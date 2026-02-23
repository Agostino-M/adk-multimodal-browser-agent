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
                return f"Navigated to {url}"
            except Exception as e:
                return f"Error navigating to {url}: {str(e)}"

    async def click(
        self,
        mode: Literal["text", "selector", "coordinates"] = "text",
        text: Optional[str] = None,
        selector: Optional[str] = None,
        coordinates: Optional[str] = None,
        exact: bool = False,
        timeout_ms: int = 10000
    ) -> Dict[str, Any]:
        """Click tool (multi-mode).

        Use cases:
        - mode="text": click an element containing `text`.
        Provide: text, optional exact.
        - mode="selector": click an element by CSS/XPath selector.
        Provide: selector.
        - mode="coordinates": click at viewport coordinates "<point>x y</point>".
        Provide: coordinates. Use only if text/selector cannot uniquely identify the element.

        Returns:
        dict with clicked info and url_after, or {"error": "..."}.
        """
        await self._ensure_started()
        async with self._page_lock:
            await self._wait_for_load_state()
            try:
                if mode == "text":
                    if not text:
                        return {"error": "mode='text' requires: text"}

                    # prova role
                    for role in ["button", "link"]:
                        loc = self.active_page.get_by_role(role, name=text, exact=exact)
                        if await loc.count() > 0:
                            await loc.first.click(timeout=timeout_ms)
                            await self._wait_for_load_state()
                            return {"clicked_mode": "text", "role": role, "text": text, "url_after": self.active_page.url}

                    # fallback text=
                    loc = self.active_page.locator(f'text="{text}"') if exact else self.active_page.locator(f"text={text}")
                    if await loc.count() == 0:
                        return {"error": f"No element found containing text: {text}"}
                    await loc.first.click(timeout=timeout_ms)
                    await self._wait_for_load_state()
                    return {"clicked_mode": "text", "role": None, "text": text, "url_after": self.active_page.url}

                if mode == "selector":
                    if not selector:
                        return {"error": "mode='selector' requires: selector"}
                    el = await self.active_page.query_selector(selector)
                    if not el:
                        return {"error": f"No element found for selector: {selector}"}
                    await el.click(timeout=timeout_ms)
                    await self._wait_for_load_state()
                    return {"clicked_mode": "selector", "selector": selector, "url_after": self.active_page.url}

                if mode == "coordinates":
                    if not coordinates:
                        return {"error": "mode='coordinates' requires: coordinates"}
                    x, y = self._parse_point(coordinates)
                    await self.active_page.mouse.click(x, y)
                    await self._wait_for_load_state()
                    return {"clicked_mode": "coordinates", "clicked_at": [x, y], "url_after": self.active_page.url}

                return {"error": f"Unknown mode: {mode}"}

            except PlaywrightTimeoutError as e:
                return {"error": "Timeout during click", "mode": mode, "details": str(e)}


    async def type(self, selector: str, content: str):
        """Types into an input field safely.
            - Scrolls into view
            - Focuses the input
            - Clears any existing text
            - Types the new content
        """
        await self._ensure_started()
        async with self._page_lock:
            print(f"[DEBUG] Typing into selector: {selector} with content: {content}")

            element = await self.active_page.query_selector(selector)
            if not element:
                return {"error": f"No element found for selector: {selector}"}

            await element.scroll_into_view_if_needed()
            await element.focus()
            await element.fill("")
            await element.type(content)

            await self._wait_for_load_state()
            return {"typed_into": selector, "content": content}
    
    async def scroll(
        self,
        mode: Literal["step", "percent", "y", "to_text", "to_selector"] = "step",
        direction: Literal["down", "up", "left", "right"] = "down",
        steps: int = 1,
        overlap: float = 0.15,
        percent: Optional[float] = None,
        y: Optional[int] = None,
        text: Optional[str] = None,
        selector: Optional[str] = None,
        smooth: bool = False,
        max_steps: int = 12,
        settle_ms: int = 300,
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

            before = await self._get_scroll_metrics()

            # Helper: clamp overlap
            ov = max(0.0, min(float(overlap), 0.5))

            async def _do_settle():
                # small wait after scroll to allow page to update (e.g. lazy load)
                if settle_ms and settle_ms > 0:
                    await self.active_page.wait_for_timeout(int(settle_ms))
                await self._wait_for_load_state()

            if mode == "percent":
                if percent is None:
                    return {"status": "error", "message": "percent is required for mode='percent'"}

                p = max(0.0, min(float(percent), 100.0))
                # y target = (docH - viewportH) * p%
                target = int((before["docH"] - before["viewportH"]) * (p / 100.0))
                target = max(0, min(target, max(0, before["docH"] - before["viewportH"])))

                await self.active_page.evaluate(
                    """({top, smooth}) => {
                        window.scrollTo({ top, left: 0, behavior: smooth ? "smooth" : "instant" });
                    }""",
                    {"top": target, "smooth": smooth},
                )
                await _do_settle()

            elif mode == "y":
                if y is None:
                    return {"status": "error", "message": "y is required for mode='y'"}

                target = int(y)
                target = max(0, min(target, max(0, before["docH"] - before["viewportH"])))

                await self.active_page.evaluate(
                    """({top, smooth}) => {
                        window.scrollTo({ top, left: 0, behavior: smooth ? "smooth" : "instant" });
                    }""",
                    {"top": target, "smooth": smooth},
                )
                await _do_settle()

            elif mode == "to_selector":
                if not selector:
                    return {"status": "error", "message": "selector is required for mode='to_selector'"}

                el = await self.active_page.query_selector(selector)
                if not el:
                    return {"status": "error", "message": f"No element found for selector: {selector}"}

                # scroll into view, then optionally align in upper portion
                await el.scroll_into_view_if_needed()
                await self.active_page.evaluate("""
                (sel) => {
                const el = document.querySelector(sel);
                if (!el) return;
                const r = el.getBoundingClientRect();
                const target = window.scrollY + r.top - (window.innerHeight * 0.2);
                window.scrollTo({ top: target, left: 0, behavior: "instant" });
                }
                """, selector)
                await _do_settle()

            elif mode == "to_text":
                if not text:
                    return {"status": "error", "message": "text is required for mode='to_text'"}

                needle = text.strip()
                if not needle:
                    return {"status": "error", "message": "text is empty"}

                # Scan loop: look for element containing text in viewport, if not found step with overlap, repeat up to max_steps.
                found = None
                attempts = 0

                while attempts < max_steps:
                    found = await self.active_page.evaluate("""
                    (needle) => {
                    needle = (needle || "").toLowerCase();

                    // look for elements containing the text (case-insensitive)
                    const candidates = Array.from(document.querySelectorAll("h1,h2,h3,h4,h5,h6,a,button,p,li,section,div"));
                    for (const node of candidates) {
                        const t = (node.innerText || node.textContent || "").trim();
                        if (t && t.toLowerCase().includes(needle)) {
                        const r = node.getBoundingClientRect();
                        const target = window.scrollY + r.top - (window.innerHeight * 0.2);
                        window.scrollTo({ top: target, left: 0, behavior: "instant" });
                        return { tag: node.tagName, text: t.slice(0, 140) };
                        }
                    }
                    return null;
                    }
                    """, needle)

                    if found:
                        break

                    # if not found, step with overlap and try again
                    metrics = await self._get_scroll_metrics()
                    if metrics["atBottom"] and direction == "down":
                        break

                    step_px = int(metrics["viewportH"] * (1.0 - ov))
                    dx = dy = 0
                    if direction == "down":
                        dy = step_px
                    elif direction == "up":
                        dy = -step_px
                    elif direction == "right":
                        dx = step_px
                    elif direction == "left":
                        dx = -step_px

                    await self.active_page.evaluate("""({dx, dy}) => window.scrollBy(dx, dy)""", {"dx": dx, "dy": dy})
                    await _do_settle()
                    attempts += 1

            else:
                # default: step
                metrics = before
                step_px = int(metrics["viewportH"] * (1.0 - ov))
                dx = dy = 0
                if direction == "down":
                    dy = step_px
                elif direction == "up":
                    dy = -step_px
                elif direction == "right":
                    dx = step_px
                elif direction == "left":
                    dx = -step_px

                steps_n = max(1, int(steps))
                for _ in range(steps_n):
                    await self.active_page.evaluate("""({dx, dy}) => window.scrollBy(dx, dy)""", {"dx": dx, "dy": dy})
                    await _do_settle()

            after = await self._get_scroll_metrics()
            anchor = await self._get_viewport_anchor()

            resp = {
                "status": "ok",
                "mode": mode,
                "direction": direction,
                "scrollY_before": before["scrollY"],
                "scrollY_after": after["scrollY"],
                "docH": after["docH"],
                "viewportH": after["viewportH"],
                "atBottom": after["atBottom"],
                "anchor": anchor,
            }

            # Include anchor info for text/selector modes to help agent continuity
            if mode == "to_text":
                resp["target_text"] = text
            if mode == "to_selector":
                resp["target_selector"] = selector

            return resp


    async def wait(self, ms: int = 5000):
        """Waits for a short period to allow the page to update."""
        await self._ensure_started()
        async with self._page_lock:
            await self.active_page.wait_for_timeout(ms)
            return f"Waited for {ms} ms"

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

            return "Closed"
