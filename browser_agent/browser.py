import os
import re
import base64
from typing import Any, Dict, Literal, Optional
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


    async def _take_screenshot(self, path: str = "screenshot.png"):
        await self.active_page.screenshot(path=path)
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
        
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
                    rect: {
                        x: rect.x,
                        y: rect.y,
                        width: rect.width,
                        height: rect.height
                    }
                };
            });
        }
        """)
        return elements

    async def get_state(self) -> dict:
        """
        Returns the full observable state of the browser.

        Includes:
        - Current page URL
        - Screenshot of the visible viewport
        - Structured list of interactive DOM elements
        """
        await self._ensure_started()
        await self._wait_for_load_state()

        img_b64 = await self._take_screenshot()
        dom = await self._extract_interactive_elements()

        return {
            "page_url": self.active_page.url,
            "page_screenshot_base64": f"data:image/png;base64,{img_b64}",
            "interactive_elements": dom,
        }
        
    async def goto_url(self, url: str):
        """Navigates to the specified URL."""
        await self._ensure_started()

        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        try:
            await self.active_page.goto(url, timeout=10000, wait_until="domcontentloaded")
            await self._wait_for_load_state()
            return f"Navigated to {url}"
        except Exception as e:
            return f"Error navigating to {url}: {str(e)}"

    '''async def click_by_coordinates(self, coordinates: str):
        """Clicks at the specified coordinates.
        Args:
            coordinates: A string in the format '<point>x y</point>' representing the point
        """
        await self._ensure_started()

        x, y = self._parse_point(coordinates)

        element = await self.active_page.evaluate(
            """([x, y]) => {
                const el = document.elementFromPoint(x, y);
                if (!el) return null;
                return {
                    tag: el.tagName,
                    id: el.id,
                    classes: el.className,
                    text: el.innerText?.slice(0, 100)
                };
            }""",
            [x, y]
        )

        print(f"[DEBUG] About to click at {x},{y}")
        print(f"[DEBUG] Element at point: {element}")

        await self.active_page.mouse.click(x, y)
        print(f"[DEBUG] Current URL after click: {self.active_page.url}")
        await self._wait_for_load_state()

        return {
            "clicked_at": [x, y],
            "element": element,
            "url_after": self.active_page.url
        }

    async def click_by_selector(self, selector: str):
        """
        Click an element using a Xpath/CSS/Playwright selector.
        Example: 'input[type="text"]', '//div[@class="myClass"]', '//button[contains(text(), "Hello")]'
        """
        await self._wait_for_load_state()

        element = await self.active_page.query_selector(selector)
        if not element:
            return {"error": f"No element found for selector: {selector}"}

        await element.click()
        await self._wait_for_load_state()

        return {
            "clicked_selector": selector,
            "url_after": self.active_page.url
        }

    async def click_by_text(self, text: str, exact: bool = False):
        """Click the first visible element containing the given text."""
        await self._ensure_started()
        await self._wait_for_load_state()

        locator = self.active_page.locator(f'text="{text}"') if exact else self.active_page.locator(f"text={text}")
        if await locator.count() == 0:
            return {"error": f"No element found containing text: {text}"}

        await locator.first.click()
        await self._wait_for_load_state()

        return {"clicked_text": text, "url_after": self.active_page.url}'''


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

        element = await self.active_page.query_selector(selector)
        if not element:
            return {"error": f"No element found for selector: {selector}"}

        await element.scroll_into_view_if_needed()
        await element.focus()
        await element.fill("")
        await element.type(content)

        await self._wait_for_load_state()
        return {"typed_into": selector, "content": content}

    '''async def scroll(self, point: str, direction: str):
        """Scrolls the page in the specified direction from the given coordinates."""
        await self._ensure_started()

        x, y = self._parse_point(point)
        await self.active_page.mouse.move(x, y)

        if direction == "down":
            await self.active_page.mouse.wheel(0, 1000)
        elif direction == "up":
            await self.active_page.mouse.wheel(0, -1000)
        elif direction == "right":
            await self.active_page.mouse.wheel(1000, 0)
        elif direction == "left":
            await self.active_page.mouse.wheel(-1000, 0)

        await self._wait_for_load_state()
        return f"Scrolled {direction} at ({x}, {y})"'''
    
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

        before = await self._get_scroll_metrics()

        # Helper: clamp overlap
        ov = max(0.0, min(float(overlap), 0.5))

        async def _do_settle():
            # piccola attesa per layout/lazy load
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

            # Scan loop (interno alla stessa chiamata tool): prova a trovare testo,
            # se non c'è, scendi con step "grandi" ma con overlap.
            found = None
            attempts = 0

            while attempts < max_steps:
                found = await self.active_page.evaluate("""
                (needle) => {
                  needle = (needle || "").toLowerCase();

                  // Cerca prima tra elementi "ragionevoli" (evita traversal infinito su DOM enormi)
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

                # se non trovato, step down (o up) con overlap
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

        # info extra se to_text / to_selector
        if mode == "to_text":
            resp["target_text"] = text
        if mode == "to_selector":
            resp["target_selector"] = selector

        return resp


    async def wait(self, ms: int = 5000):
        """Waits for a short period to allow the page to update."""
        await self._ensure_started()
        await self.active_page.wait_for_timeout(ms)
        return f"Waited for {ms} ms"

    async def close(self):
        """Closes the browser and cleans up resources."""
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
