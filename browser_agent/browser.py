import re
import json
import asyncio
import aiofiles
from google.genai import types
from typing import Any, Dict, List, Literal, Optional
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright


class BrowserManager:
    def __init__(self, show_browser: bool = True):
        self.show_browser = show_browser
        self.playwright = None
        self.driver = None
        self.context = None
        self.active_page = None
        self._started = False
        self._browser_lock = asyncio.Lock()
        self._page_lock = asyncio.Lock()

    async def init(self):
        if self._started:
            return

        self.playwright = await async_playwright().start()
        self.driver = await self.playwright.chromium.launch(headless=not self.show_browser)

        self.context = await self.driver.new_context(
            record_video_dir="videos/",
            record_video_size={"width": 1280, "height": 720}
        )

        self.active_page = await self.context.new_page()
        self.active_page.set_default_timeout(10000)
        self._started = True

    async def _ensure_started(self):
        """Lazy init: start browser on first tool call."""
        if not self._started:
            async with self._browser_lock:
                if not self._started: # Double check locking
                    await self.init()

    async def _wait_for_load_state(self):
        """Waits for the page to be idle after an action."""
        try:
            await self.active_page.wait_for_load_state("networkidle", timeout=5000)
        except:
            pass

    def _parse_point(self, point: str):
        # support both <point>x y</point> and HTML-escaped &lt;point&gt;x y&lt;/point&gt;
        clean = (
            point.replace("&lt;point&gt;", "")
                 .replace("&lt;/point&gt;", "")
                 .replace("<point>", "")
                 .replace("</point>", "")
                 .strip()
        )
        numbers = re.findall(r"-?\d+", clean)

        if len(numbers) < 2:
            raise ValueError(f"Invalid point format: {point}")

        return int(numbers[0]), int(numbers[1])

    async def _extract_interactive_elements(self):
        elements = await self.active_page.evaluate("""
        () => {
            const els = Array.from(document.querySelectorAll(
                'a, button, input, textarea, select, [role="button"]'
            ));
            return els.map((el, index) => {
                const rect = el.getBoundingClientRect();
                return {
                    id: index,
                    tag: el.tagName,
                    text: el.innerText?.slice(0, 100),
                    aria: el.getAttribute("aria-label"),
                    rect: { x: rect.x, y: rect.y, width: rect.width, height: rect.height }
                };
            });
        }
        """)
        return elements

    def _sanitize_css_selector(self, selector: str) -> str:
        selector = selector.strip()

        # escape quotes inside attribute values
        selector = re.sub(r"(?<=\=)['\"]([^'\"]*)['\"]", lambda m: '"' + m.group(1).replace('"', '\\"') + '"', selector)

        # remove potentially dangerous characters that could break the selector
        selector = re.sub(r"[^a-zA-Z0-9\[\]=\.\#\*\-_\s\>\+\~]", "", selector)

        return selector

    async def get_state(self) -> List[types.Part]:
        """
        Returns the full observable state of the browser.

        Includes:
        - Current page URL
        - Screenshot of the visible viewport
        - Structured list of interactive DOM elements
        """
        await self._ensure_started()
        async with self._page_lock:
            await self._wait_for_load_state()

            await self.active_page.screenshot(path="screenshot.jpg", type="jpeg", quality=60)
            async with aiofiles.open("screenshot.jpg", "rb") as f:
                image_bytes = await f.read()

            dom = await self._extract_interactive_elements()

            structured_state = {
                "page_url": self.active_page.url,
                "interactive_elements": dom[:40]
            }
            return [
                types.Part.from_text(text=json.dumps(structured_state)),
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
            ]

    async def goto_url(self, url: str):
        """Navigates to the specified URL."""
        await self._ensure_started()
        async with self._page_lock:
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
            try:
                await self.active_page.goto(url, timeout=10000, wait_until="domcontentloaded")
                await self._wait_for_load_state()
                return {"status": "success", "url": self.active_page.url}
            except Exception as e:
                return {"status": "error", "message": f"Error navigating to {url}: {str(e)}"}

    async def _click_by_text(
        self, text: Optional[str], exact: bool, timeout_ms: int
    ) -> Dict[str, Any]:
        if not text:
            return {"status": "error", "message": "mode='text' requires: text"}

        for role in ["button", "link"]:
            loc = self.active_page.get_by_role(role, name=text, exact=exact)
            if await loc.count() > 0:
                await loc.first.click(timeout=timeout_ms)
                await self._wait_for_load_state()
                return {"status": "success", "clicked_mode": "text", "role": role, "text": text, "url_after": self.active_page.url}

        # fallback text locator
        loc = self.active_page.locator(f"text={text}")
        if await loc.count() == 0:
            # fuzzy fallback: primi 10 caratteri
            loc = self.active_page.locator(f"text={text[:10]}")
            if await loc.count() == 0:
                return {"status": "error", "message": f"No element found containing text: {text}"}

        await loc.first.click(timeout=timeout_ms)
        await self._wait_for_load_state()
        return {"status": "success", "clicked_mode": "text", "role": None, "text": text, "url_after": self.active_page.url}

    async def _click_by_selector(
        self, selector: Optional[str], timeout_ms: int
    ) -> Dict[str, Any]:
        if not selector:
            return {"status": "error", "message": "mode='selector' requires: selector"}

        try:
            el = await self.active_page.query_selector(self._sanitize_css_selector(selector))
        except Exception as e:
            print(f"[ERROR] Selector query failed: {selector}, error: {e}")
            return {"status": "error", "message": f"Selector query failed: {e}"}
        
        if not el:
            return {"status": "error", "message": f"No element found for selector: {selector}"}

        await el.click(timeout=timeout_ms)
        await self._wait_for_load_state()
        return {"status": "success", "clicked_mode": "selector", "selector": selector, "url_after": self.active_page.url}

    async def _click_by_coordinates(
        self, coordinates: Optional[str]
    ) -> Dict[str, Any]:
        if not coordinates:
            return {"status": "error", "message": "mode='coordinates' requires: coordinates"}
        x, y = self._parse_point(coordinates)
        await self.active_page.mouse.click(x, y)
        await self._wait_for_load_state()
        return {"status": "success", "clicked_mode": "coordinates", "clicked_at": [x, y], "url_after": self.active_page.url}

    async def click(
        self,
        mode: Literal["text", "selector", "coordinates"] = "text",
        text: Optional[str] = None,
        selector: Optional[str] = None,
        coordinates: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Click tool (multi-mode).

        Use cases:
        - mode="text": click an element containing `text`.
        Provide: text
        - mode="selector": click an element by CSS/XPath selector.
        Provide: selector.
        - mode="coordinates": click at viewport coordinates "<point>x y</point>".
        Provide: coordinates. Use only if text/selector cannot uniquely identify the element.

        Returns:
        dict with clicked info and url_after, or {"error": "..."}.
        """
        timeout_ms = 10000

        await self._ensure_started()
        async with self._page_lock:
            await self._wait_for_load_state()
            try:
                if mode == "text":
                    return await self._click_by_text(text, exact=True, timeout_ms=timeout_ms)

                if mode == "selector":
                    return await self._click_by_selector(selector, timeout_ms=timeout_ms)

                if mode == "coordinates":
                    return await self._click_by_coordinates(coordinates)                        

                return {"status": "error", "message": f"Unknown mode: {mode}"}

            except PlaywrightTimeoutError as e:
                return {"status": "error", "message": "Timeout during click", "mode": mode, "details": str(e)}

    async def type(self, selector: str, content: str):
        """Types into an input field safely."""
        await self._ensure_started()
        async with self._page_lock:
            print(f"[DEBUG] Typing into selector: {selector} with content: {content}")

            element = await self.active_page.query_selector(selector)
            if not element:
                return {"status": "error", "message": f"No element found for selector: {selector}"}

            await element.scroll_into_view_if_needed()
            await element.focus()
            await element.fill("")
            await element.type(content)

            await self._wait_for_load_state()
            return {"status": "success", "typed_into": selector, "content": content}

    async def _get_scroll_metrics(self) -> Dict[str, Any]:
        """
        Returns basic scroll metrics for the current page.
        """
        metrics = await self.active_page.evaluate("""
            () => {
                const scrollY = window.scrollY || window.pageYOffset;
                const viewportH = window.innerHeight;
                const docH = Math.max(
                    document.body.scrollHeight,
                    document.documentElement.scrollHeight,
                    document.body.offsetHeight,
                    document.documentElement.offsetHeight,
                    document.body.clientHeight,
                    document.documentElement.clientHeight
                );
                return {
                    scrollY,
                    viewportH,
                    docH,
                    atBottom: scrollY + viewportH >= docH - 2  // small tolerance
                };
            }
        """)
        return metrics

    async def scroll_percent(
        self, percent: float, before: Dict[str, Any]
    ) -> Dict[str, Any]:
        if percent is None:
            return {"status": "error", "message": "percent required"}
        target = int((before["docH"] - before["viewportH"]) * (percent / 100))
        await self.active_page.evaluate(
            """({top}) => window.scrollTo({ top, left: 0 })""", {"top": target}
        )

    async def scroll_y(self, y: int, before: Dict[str, Any]) -> Dict[str, Any]:
        if y is None:
            return {"status": "error", "message": "y required"}
        target = max(0, min(y, before["docH"] - before["viewportH"]))
        await self.active_page.evaluate(
            """({top}) => window.scrollTo({ top, left: 0 })""", {"top": target}
        )

    async def scroll_to_selector(self, selector: str) -> Dict[str, Any]:
        if not selector:
            return {"status": "error", "message": "selector required"}
        try:
            el = await self.active_page.query_selector(self._sanitize_css_selector(selector))
        except Exception as e:
            logging.warning(f"Selector query failed: {selector}, error: {e}")
            return {"status": "error", "message": f"Selector query failed: {e}"}

        if not el:
            return {
                "status": "error",
                "message": f"No element for selector: {selector}",
            }
        await el.scroll_into_view_if_needed()

    async def scroll_to_text(self, text: str) -> Dict[str, Any]:
        if not text:
            return {"status": "error", "message": "text required"}
        needle = text.strip().lower()
        found = await self.active_page.evaluate(
            """(needle) => {
                const elems = Array.from(document.querySelectorAll("h1,h2,h3,h4,h5,h6,a,button,p,li,section,div"));
                for (const e of elems) {
                    if ((e.innerText || "").toLowerCase().includes(needle)) {
                        e.scrollIntoView({ block: "center" });
                        return e.innerText.slice(0, 140);
                    }
                }
                return null;
            }""",
            needle,
        )
        if not found:
            return {"status": "error", "message": f"Text not found: {text}"}

    async def scroll_step(
        self,
        direction: Literal["down", "up", "left", "right"],
        steps: int,
        before: Dict[str, Any],
    ) -> Dict[str, Any]:
        step_px = int(before["viewportH"] * 0.8)
        dx = dy = 0
        if direction == "down":
            dy = step_px
        elif direction == "up":
            dy = -step_px
        elif direction == "right":
            dx = step_px
        elif direction == "left":
            dx = -step_px
        for _ in range(max(1, steps)):
            await self.active_page.evaluate("""({dx, dy}) => window.scrollBy(dx, dy)""", {"dx": dx, "dy": dy})


    async def scroll(
        self,
        mode: Literal["step", "percent", "y", "to_text", "to_selector"] = "step",
        direction: Literal["down", "up", "left", "right"] = "down",
        steps: int = 1,
        percent: Optional[float] = None,
        y: Optional[int] = None,
        text: Optional[str] = None,
        selector: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Multi-mode scroll tool.

        Modes:
        - step: scroll in small steps with overlap to avoid skipping content.
                Uses direction + steps + overlap.
        - percent: jump to a percentage of the document (0..100). Uses percent.
        - y: jump to absolute Y position in pixels. Uses y.
        - to_text: find first element containing `text` (case-insensitive) and scroll it into view.
                   Uses text. Internally may scan by stepping up to max_steps.
        - to_selector: scroll element matching CSS selector into view. Uses selector.

        Returns metrics (before/after), and an anchor snippet for continuity.
        """
        await self._ensure_started()
        async with self._page_lock:

            async def _settle():
                # small wait after scroll
                await self.active_page.wait_for_timeout(300)

            # get basic metrics
            before = await self._get_scroll_metrics()

            if mode == "percent":
                await self.scroll_percent(percent, before)

            elif mode == "y":
                await self.scroll_y(y, before)

            elif mode == "to_selector":
                await self.scroll_to_selector(selector)

            elif mode == "to_text":
                await self.scroll_to_text(text)

            else:  # step
                await self.scroll_step(direction, steps, before)

            await _settle()
            after = await self._get_scroll_metrics()

            return {
                "status": "ok",
                "mode": mode,
                "scrollY_before": before["scrollY"],
                "scrollY_after": after["scrollY"],
                "docH": after["docH"],
                "viewportH": after["viewportH"],
                "atBottom": after["atBottom"],
            }

    async def wait(self, ms: int = 5000):
        """Waits for a short period to allow the page to update."""
        await self._ensure_started()
        async with self._page_lock:
            await self.active_page.wait_for_timeout(ms)
            return {"status": "success", "waited_ms": ms}

    async def close(self):
        """Closes the browser and cleans up resources."""
        async with self._page_lock:
            try:
                if self.context:
                    await self.context.close()
                if self.driver:
                    await self.driver.close()
                if self.playwright:
                    await self.playwright.stop()
            finally:
                self.context = None
                self.driver = None
                self.playwright = None
                self.active_page = None
                self._started = False

            return {"status": "success", "message": "Browser closed"}
        
    async def press_key(
        self,
        keys: List[str],
    ) -> Dict[str, Any]:
        """
        Simulate pressing one or more keyboard keys in sequence.

        Args:
            keys: A list of key names to press in order.
                Common keys: "Enter", "Tab", "ArrowDown", "ArrowUp",
                "ArrowLeft", "ArrowRight", "Escape", "Backspace"
                For combinations, separate keys in order:
                e.g. ["Control", "A"] for Ctrl+A.

        Returns:
            A dict describing what was pressed.
        """
        await self._ensure_started()
        async with self._page_lock:
            await self._wait_for_load_state()

            try:
                for key in keys:
                    await self.active_page.keyboard.press(key)

                await self._wait_for_load_state()

                return {"status": "success", "pressed_keys": keys, "url_after": self.active_page.url}
            except Exception as e:
                return {"status": "error", "message": f"Keyboard press failed: {str(e)}", "pressed_keys": keys}
