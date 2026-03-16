import logging
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

        # RAG helper for DOM elements. We instantiate once to avoid
        # reloading the transformer on every request.
        self._dom_retriever = DOMRetriever()

    async def init(self):
        logging.info(f"Initializing browser: {self._started}")
        if self._started:
            return

        self.playwright = await async_playwright().start()
        self.driver = await self.playwright.chromium.launch(headless=not self.show_browser)

        self.context = await self.driver.new_context(
            viewport={"width": 1024, "height": 768},
            record_video_dir="videos/",
            record_video_size={"width": 1024, "height": 768},
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

    async def _extract_interactive_elements(self, limit: int = 50):
        elements = await self.active_page.evaluate(
            """
            (limit) => {
            const els = Array.from(document.querySelectorAll(
                'a, button, input, textarea, select, [role="button"]'
            ));

            const clean = (s) => (s || '').replace(/\\s+/g, ' ').trim(); // normalize whitespace

            const getText = (el) => {
                const aria = el.getAttribute("aria-label");
                if (aria) return aria;

                const tag = el.tagName.toLowerCase();

                if (tag === "input" || tag === "textarea") {
                const type = (el.getAttribute("type") || "").toLowerCase();
                const ph = el.getAttribute("placeholder") || "";
                if (type === "password") return ph;
                return ph || el.value || "";
                }

                if (tag === "select") {
                const opt = el.selectedOptions && el.selectedOptions[0];
                return (opt && (opt.innerText || opt.textContent)) || "";
                }

                return el.innerText || el.textContent || "";
            };

            const out = els.map((el) => {
                const rect = el.getBoundingClientRect();
                if (rect.width <= 1 || rect.height <= 1) return null;

                const style = window.getComputedStyle(el);
                if (!style) return null;
                if (style.display === 'none' || style.visibility === 'hidden') return null;
                if (style.pointerEvents === 'none') return null;
                if (Number(style.opacity) === 0) return null;

                if (el.hasAttribute("disabled")) return null;
                if (el.getAttribute("aria-disabled") === "true") return null;

                const text = clean(getText(el)).slice(0, 100);

                const tag = el.tagName.toLowerCase();
                if (!text && !["input", "textarea", "select"].includes(tag)) return null; // filter out non-interactive elements without text

                let result = `-${tag}: ${text}`;
                
                const attrs = [];
                if (el.id) attrs.push(`id=${el.id}`);
                const nameAttr = el.getAttribute("name");
                if (nameAttr) attrs.push(`name=${nameAttr}`);
                const ariaAttr = el.getAttribute("aria-label");
                if (ariaAttr) attrs.push(`aria=${ariaAttr}`);
                
                if (attrs.length > 0) {
                    result += ` (${attrs.join(', ')})`;
                }

                return result;                
            }).filter(Boolean);

            return out.slice(0, limit);
            }
            """,
            limit,
        )
        return elements

    async def _retrieve_relevant_elements(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Tool that performs a semantic search over the current DOM.

        The agent can call this when it has a ``CURRENT TASK`` and wants to
        narrow the list of interactive elements to those that appear
        semantically relevant. ``query`` is usually the task description.
        """
        await self._ensure_started()
        start = time.time()
        # pull the latest elements from the page
        elements = await self._extract_interactive_elements(limit=1000)
        c1 = time.time()
        logging.info(f"DOM extraction got {len(elements)} elements in {c1 - start:.2f} seconds for query: \"{query}\"")
        if not query:
            return elements[:k]  # if no query, just return the first k elements

        self._dom_retriever.build_index(elements)
        c2 = time.time()
        logging.info(f"DOM index built with {len(elements)} elements for query: \"{query}\" in {c2 - c1:.2f} seconds")
        results = self._dom_retriever.query(query, k)
        c3 = time.time()
        logging.info(f"DOM retrieval for query: \"{query}\" returned {len(results)} results in {c3 - c2:.2f} seconds")
        return results

    async def get_state(self, query: str = "", with_screenshot: bool = True) -> List[types.Part]:
        """
        Returns the full observable state of the browser.
        ``query`` is an optional string that can be used to filter the DOM elements using the RAG tool before returning the state.
        ``with_screenshot`` controls whether to include a screenshot of the current page in the returned state.

        Includes:
        - Current page URL
        - Screenshot of the visible viewport (only if with_screenshot=True)
        - Structured list of interactive DOM elements from the current page, optionally filtered by relevance to the query.
        """
        await self._ensure_started()

        async with self._page_lock:
            await self._wait_for_load_state()

            #dom = await self._extract_interactive_elements(40) old version without rag
            dom = await self._retrieve_relevant_elements(query=query, k=30)  # new version with RAG filtering
            logging.info(f"Retrieved {len(dom)} relevant DOM elements for state query: \"{query}\"")
            # Compact custom format to save tokens
            lines = [f"url: {self.active_page.url}"]
            if dom:
                lines.append("elements:")
                lines.extend(dom)  # dom now contains pre-formatted strings
            else:
                lines.append("elements: none")

            compact_text = "\n".join(lines)

            state = [
                types.Part.from_text(text=compact_text),
            ]

            if with_screenshot:
                await self.active_page.screenshot(path="screenshot.jpg", type="jpeg", quality=60)
                async with aiofiles.open("screenshot.jpg", "rb") as f:
                    image_bytes = await f.read()
                    
                state.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))

            return state

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
                clicked_text = await loc.first.inner_text()
                return {"status": "success", "clicked_mode": "text", "role": role, "text": clicked_text, "url_after": self.active_page.url}

        # fallback text locator
        loc = self.active_page.get_by_text(text, exact=exact)
        if await loc.count() == 0:
            return {"status": "error", "message": f"No element found containing text: {text}"}

        await loc.first.click(timeout=timeout_ms)
        await self._wait_for_load_state()
        clicked_text = await loc.first.inner_text()
        role = await loc.first.get_attribute("role") or "unknown"
        return {"status": "success", "clicked_mode": "text", "role": role, "text": clicked_text, "url_after": self.active_page.url}

    async def _click_by_selector(
        self, selector: Optional[str], timeout_ms: int
    ) -> Dict[str, Any]:
        if not selector:
            return {"status": "error", "message": "mode='selector' requires: selector"}

        try:
            el = await self.active_page.query_selector(self._sanitize_css_selector(selector))
        except Exception as e:
            logging.error(f"Selector query failed: {selector}, error: {e}")
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
        clicked_text = await self.active_page.evaluate("""
            ({x, y}) => {
                const el = document.elementFromPoint(x, y);
                return el ? (el.innerText || el.textContent || "").slice(0, 100) : "";
            }
        """, {"x": x, "y": y})
        await self._wait_for_load_state()
        return {"status": "success", "clicked_mode": "coordinates", "clicked_at": [x, y], "text": clicked_text, "url_after": self.active_page.url}

    async def click(
        self,
        mode: Literal["text", "selector", "coordinates"] = "text",
        text: Optional[str] = None,
        exact: bool = True,
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
                    return await self._click_by_text(text, exact=exact, timeout_ms=timeout_ms)

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
            logging.info(f"Typing into selector: {selector} with content: {content}")

            try:
                el = await self.active_page.query_selector(selector)
            except Exception as e:
                logging.warning(f"Selector query failed: {selector}, error: {e}")
                return {"status": "error", "message": f"Selector query failed: {e}"}

            if not el:
                return {"status": "error", "message": f"No element found for selector: {selector}"}

            await el.scroll_into_view_if_needed()
            await el.focus()
            await el.fill("")
            await el.type(content)

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

    async def close(self):
        """Closes the browser and cleans up resources."""
        async with self._browser_lock:
            async with self._page_lock:
                try:
                    # Close in proper order: page -> context -> driver -> playwright
                    if self.active_page:
                        try:
                            await self.active_page.close()
                        except:
                            pass
                    if self.context:
                        try:
                            await self.context.close()
                        except:
                            pass
                    if self.driver:
                        try:
                            await self.driver.close()
                        except:
                            pass
                    if self.playwright:
                        try:
                            await self.playwright.stop()
                        except:
                            pass
                finally:
                    self.context = None
                    self.driver = None
                    self.playwright = None
                    self.active_page = None
                    self._started = False
        
        return {"status": "success", "message": "Browser closed"}
