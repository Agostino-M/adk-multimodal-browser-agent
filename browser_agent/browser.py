import logging
import re
import asyncio
import time
import aiofiles
from google.genai import types
from typing import Any, Dict, List, Literal, Optional
from playwright.async_api import Locator
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

from browser_agent.dom_retriever import DOMRetriever


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
        self._new_pages_opened = 0

        # RAG helper for DOM elements. We instantiate once to avoid
        # reloading the transformer on every request.
        self._dom_retriever = DOMRetriever()

    async def init(self):
        logging.info(f"Initializing browser: {self._started}")
        if self._started:
            return

        # Clean up any stale resources from a previous crashed session to avoid leaks.
        for obj, method in [(self.context, "close"), (self.driver, "close"), (self.playwright, "stop")]:
            if obj is not None:
                try:
                    await getattr(obj, method)()
                except Exception:
                    pass
        self.context = None
        self.driver = None
        self.playwright = None
        self.active_page = None

        self.playwright = await async_playwright().start()
        self.driver = await self.playwright.chromium.launch(headless=not self.show_browser)

        self.context = await self.driver.new_context(
            viewport={"width": 1024, "height": 768},
            record_video_dir="videos/",
            record_video_size={"width": 1024, "height": 768},
        )

        # Track new pages/tabs so the agent can decide whether to switch.
        self.context.on("page", self._handle_new_page)

        self.active_page = await self.context.new_page()
        self.active_page.set_default_timeout(10000)
        
        # Pre-load the SentenceTransformer model in a background thread
        # to avoid latency (~30s) on first DOM retrieval query
        logging.info("Pre-loading SentenceTransformer model for DOM retrieval...")
        await asyncio.to_thread(self._dom_retriever._ensure_model_loaded)
        logging.info("SentenceTransformer model loaded successfully")
        
        self._started = True

    async def _ensure_started(self):
        """Lazy init: start browser on first tool call. Also recovers from unexpected disconnections."""
        is_dead = self._started and (self.driver is None or not self.driver.is_connected())
        if not self._started or is_dead:
            async with self._browser_lock:
                is_dead = self._started and (self.driver is None or not self.driver.is_connected())
                if not self._started or is_dead:
                    if is_dead:
                        logging.warning("Browser disconnected unexpectedly, re-initializing...")
                        self._started = False
                        self.active_page = None
                        self.context = None
                        self.driver = None
                    await self.init()

    async def _wait_for_load_state(self):
        """Waits for the page to be idle after an action."""
        if self.active_page is None:
            return
        try:
            await self.active_page.wait_for_load_state("networkidle", timeout=5000)
        except Exception:
            pass

    async def _handle_new_page(self, page):
        """Handle new pages (popups/tabs) by recording them for later inspection."""
        try:
            url = page.url
        except Exception:
            url = ""
        logging.info(f"New page opened: {url}. Tracking it for page switching.")
        self._new_pages_opened += 1

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
   
    async def _resolve_locator(
        self,
        selector: str,
        *,
        require_enabled: bool = True,
        require_editable: bool = False,
    ) -> Optional[Locator]:
        """Pick the first node matching selector that passes visibility/interaction checks."""
        try:
            root = self.active_page.locator(selector)
            n = await root.count()
        except Exception as e:
            logging.warning("locator(%r) failed: %s", selector, e)
            return None
        
        for i in range(n):
            cand = root.nth(i)
            try:
                if not await cand.is_visible():
                    continue
                if require_enabled and not await cand.is_enabled():
                    continue
                if require_editable and not await cand.is_editable():
                    continue
            except Exception:
                continue
            return cand
        return None
 
    async def _scroll_into_view(self, loc: Locator, log_selector: str) -> None:
        try:
            await loc.scroll_into_view_if_needed(timeout=3000)
        except PlaywrightTimeoutError:
            logging.warning(
                "scroll_into_view_if_needed timed out for selector=%r; using JS scroll fallback",
                log_selector,
            )
            try:
                await loc.evaluate(
                    """e => e.scrollIntoView({ block: 'center', inline: 'nearest' })"""
                )
            except Exception:
                pass
 
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

                let result = `-${tag}: text=${text}`;
                
                const attrs = [];
                if (el.id) attrs.push(`id=${el.id}`);
                const nameAttr = el.getAttribute("name");
                if (nameAttr) attrs.push(`name=${nameAttr}`);
                const ariaAttr = el.getAttribute("aria-label");
                if (ariaAttr) attrs.push(`aria=${ariaAttr}`);
                
                if (attrs.length > 0) {
                    result += `, ${attrs.join(', ')}`;
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

    async def _detect_captcha(self) -> Optional[str]:
        """
        Checks the current page for CAPTCHA or anti-bot challenges.
        Returns a short label if detected, None otherwise.
        Uses URL, page title, and a small set of known DOM selectors.
        """
        try:
            url = self.active_page.url.lower()
            title = (await self.active_page.title()).lower()

            if "challenges.cloudflare.com" in url or "/cdn-cgi/challenge-platform" in url:
                return "Cloudflare challenge page"
            if "captcha" in url:
                return "CAPTCHA detected in URL"

            captcha_titles = [
                ("just a moment", "Cloudflare 'Just a moment' challenge"),
                ("verify you are human", "human-verification page"),
                ("robot or human", "robot-check page"),
                ("security check", "security-check page"),
                ("are you a robot", "robot-check page"),
                ("ddos protection", "DDoS-protection page"),
                ("robot check", "robot-check page"),
                ("please enable javascript", "JS-challenge page"),
            ]
            for keyword, label in captcha_titles:
                if keyword in title:
                    return label

            # Selectors that are always blocking when present
            blocking_selectors = [
                ("#challenge-form", "Cloudflare challenge form"),
                ("#challenge-running", "Cloudflare challenge running"),
                ("[data-hcaptcha-widget-id]", "hCaptcha widget"),
                ("input[name='captcha']", "CAPTCHA input field"),
            ]
            for selector, label in blocking_selectors:
                try:
                    if await self.active_page.query_selector(selector):
                        return label
                except Exception:
                    continue

            # reCAPTCHA and hCaptcha iframes can be invisible background widgets on normal pages.
            # Only flag them if the element is actually visible (bounding box with meaningful size).
            visible_selectors = [
                ("iframe[src*='hcaptcha']", "hCaptcha"),
                ("div.g-recaptcha", "Google reCAPTCHA widget"),
            ]
            for selector, label in visible_selectors:
                try:
                    el = await self.active_page.query_selector(selector)
                    if el:
                        box = await el.bounding_box()
                        if box and box["width"] > 100 and box["height"] > 50:
                            return label
                except Exception:
                    continue

            return None
        except Exception:
            return None

    async def _take_screenshot_with_size_limit(self, full_page: bool, metrics: Dict[str, Any]):
        """
        Takes a screenshot and ensures it stays within size limits by clipping height if needed.

        full_page=True: captures a larger region starting from the current scroll position,
        useful when the agent needs to inspect visual content. Clips downward from scrollY.
        full_page=False: captures only the current viewport.
        """
        if full_page:
            max_bytes = 400000  # 400KB for full-page mode
            scroll_y = int(metrics.get("scrollY", 0))
            viewport_w = int(metrics.get("viewportW", 1024))
            viewport_h = int(metrics["viewportH"])
            # Remaining page height from current scroll position
            remaining_h = max(int(metrics["docH"]) - scroll_y, viewport_h)

            # Empiric: ~750px per 100KB at quality=55
            safe_height = int((max_bytes / 100000) * 750)
            safe_height = min(safe_height, remaining_h)
            safe_height = max(safe_height, viewport_h)

            logging.info(
                f"Full screenshot: scrollY={scroll_y}, docH={metrics['docH']}, "
                f"remaining={remaining_h}, target_height={safe_height}px"
            )

            height_steps = [safe_height, int(safe_height * 0.8), int(safe_height * 0.6), viewport_h]

            for clip_height in height_steps:
                if clip_height < viewport_h:
                    continue
                try:
                    screenshot_path = "screenshot_full.jpg"
                    clip_region = {"x": 0, "y": scroll_y, "width": viewport_w, "height": clip_height}
                    await self.active_page.screenshot(
                        path=screenshot_path, full_page=True, type="jpeg", quality=55, clip=clip_region
                    )
                    async with aiofiles.open(screenshot_path, "rb") as f:
                        image_bytes = await f.read()
                    file_size = len(image_bytes)
                    if file_size <= max_bytes:
                        logging.info(f"Full screenshot: scrollY={scroll_y}, height={clip_height}px, size={file_size} bytes")
                        return screenshot_path, image_bytes, "full"
                    logging.info(f"Full screenshot: height={clip_height}px, size={file_size} bytes (exceeds limit, retrying)")
                except Exception as e:
                    logging.error(f"Error taking full screenshot at height {clip_height}: {e}")
                    continue

            logging.warning("Full screenshot exceeded size limit. Falling back to viewport.")
            return await self._take_screenshot_with_size_limit(full_page=False, metrics=metrics)

        else:
            # Viewport screenshot
            try:
                screenshot_path = "screenshot_viewport.jpg"
                logging.info(f"Taking viewport screenshot (viewportH={metrics['viewportH']}px)")
                await self.active_page.screenshot(path=screenshot_path, full_page=False, type="jpeg", quality=40)
                async with aiofiles.open(screenshot_path, "rb") as f:
                    image_bytes = await f.read()
                logging.info(f"Viewport screenshot: size={len(image_bytes)} bytes")
                return screenshot_path, image_bytes, "viewport"
            except Exception as e:
                logging.error(f"Error taking viewport screenshot: {e}")
                return None, None, "viewport"

    async def get_state(self, query: str = "", with_screenshot: bool = True, full_page_screenshot: bool = False) -> List[types.Part]:
        """
        Returns the full observable state of the browser.
        ``query`` is an optional string used to filter DOM elements by relevance.
        ``with_screenshot`` controls whether to include a screenshot.
        ``full_page_screenshot`` when True captures a larger region starting from the current
        scroll position (useful for inspecting visual content). When False captures the viewport only.

        Includes:
        - Current page URL
        - Screenshot of the visible viewport or enlarged region anchored at current scroll position
        - Structured list of interactive DOM elements from the current page, optionally filtered by relevance to the query.
        """
        await self._ensure_started()

        async with self._page_lock:
            try:
                await self._wait_for_load_state()

                captcha_label = await self._detect_captcha()

                #dom = await self._extract_interactive_elements(40) old version without rag
                dom = await self._retrieve_relevant_elements(query=query, k=30)  # new version with RAG filtering
                logging.info(f"Retrieved {len(dom)} relevant DOM elements for state query: \"{query}\"")

                # Get scroll metrics using the custom function _get_scroll_metrics
                metrics = await self._get_scroll_metrics()

                # Calculate the visible percentage of the page
                visible_percentage = (metrics["viewportH"] / max(metrics["docH"], 1)) * 100
                scroll_position = metrics["scrollY"]

                # Build page overview for all open tabs/pages
                pages = self.context.pages
                page_lines = [f"pages_open: {len(pages)}"]
                for idx, page in enumerate(pages):
                    try:
                        title = await page.title()
                    except Exception:
                        title = ""
                    active_flag = "(active)" if page == self.active_page else ""
                    page_lines.append(f"  {idx}: {active_flag} {title or 'no title'} | {page.url}")

                # Compact custom format to save tokens
                lines = [f"url: {self.active_page.url}"]
                if captcha_label:
                    logging.warning(f"CAPTCHA detected: {captcha_label}")
                    lines.insert(0, f"CAPTCHA_DETECTED: {captcha_label} — this page requires human verification. Stop interacting with this page and report the blocker to the planner.")
                lines.append(f"visible_percentage: {visible_percentage:.2f}% of the page visible in the viewport.")
                lines.append(f"scroll_position: {scroll_position:.2f} pixels down the page.")
                lines.extend(page_lines)
                
                if dom:
                    lines.append("elements:")
                    lines.extend(dom)  # dom now contains pre-formatted strings
                else:
                    lines.append("elements: none")

                compact_text = "\n".join(lines)

                state = [
                    types.Part.from_text(text=compact_text),
                ]

                # Screenshot part - with size control
                if with_screenshot:
                    image_bytes = None
                    screenshot_type = "viewport"

                    _, image_bytes, screenshot_type = await self._take_screenshot_with_size_limit(
                        full_page=full_page_screenshot,
                        metrics=metrics
                    )
                    
                    if image_bytes:
                        logging.info(f"Screenshot taken ({screenshot_type}): {len(image_bytes)} bytes")
                        state.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
                    else:
                        logging.warning("Screenshot creation failed or exceeded size limits")

            except Exception as e:
                logging.error(f"Error in get_state: {e}")
                state = [types.Part.from_text(text=f"Error retrieving state: {str(e)}")]
        
        return state

    async def goto_url(self, url: str):
        """Navigates to the specified URL."""
        await self._ensure_started()
        async with self._page_lock:
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
            self._new_pages_opened = 0  # Reset counter before action
            try:
                await self.active_page.goto(url, timeout=10000, wait_until="domcontentloaded")
                await self._wait_for_load_state()
                result = {"status": "success", "url": self.active_page.url}
                if self._new_pages_opened > 0:
                    result["new_pages_opened"] = self._new_pages_opened
                return result
            except Exception as e:
                result = {"status": "error", "message": f"Error navigating to {url}: {str(e)}"}
                if self._new_pages_opened > 0:
                    result["new_pages_opened"] = self._new_pages_opened
                return result

    async def _click_by_text(
        self, text: Optional[str], exact: bool, timeout_ms: int
    ) -> Dict[str, Any]:
        if not text:
            return {"status": "error", "message": "mode='text' requires: text"}
 
        # Prefer accessible name (includes aria-label). Covers combobox / search fields
        # where get_by_text fails because there is no visible text node.
        for role in ("button", "link", "textbox", "searchbox", "combobox"):
            loc = self.active_page.get_by_role(role, name=text, exact=exact)
            if await loc.count() > 0:
                clicked_text = await loc.first.inner_text()
                await loc.first.click(timeout=timeout_ms)
                await self._wait_for_load_state()
                return {"status": "success", "clicked_mode": "text", "role": role, "text": clicked_text, "url_after": self.active_page.url}
 
        loc = self.active_page.get_by_label(text, exact=exact)
        if await loc.count() > 0:
            clicked_text = await loc.first.inner_text()
            role = await loc.first.get_attribute("role") or "label"
            await loc.first.click(timeout=timeout_ms)
            await self._wait_for_load_state()
            return {"status": "success", "clicked_mode": "text", "role": role, "text": clicked_text, "url_after": self.active_page.url}
 
        # fallback: visible text / subtree text (does not match aria-label-only inputs)
        loc = self.active_page.get_by_text(text, exact=exact)
        if await loc.count() == 0:
            return {"status": "error", "message": f"No element found containing text: {text}"}
 
        clicked_text = await loc.first.inner_text()
        await loc.first.click(timeout=timeout_ms)
        await self._wait_for_load_state()
        role = await loc.first.get_attribute("role") or "unknown"
        return {"status": "success", "clicked_mode": "text", "role": role, "text": clicked_text, "url_after": self.active_page.url}
 
    async def _click_by_selector(
        self, selector: Optional[str], timeout_ms: int
    ) -> Dict[str, Any]:
        if not selector:
            return {"status": "error", "message": "mode='selector' requires: selector"}

        try:
            loc = await self._resolve_locator(selector, require_enabled=True, require_editable=False)
        except Exception as e:
            logging.error(f"Selector query failed: {selector}, error: {e}")
            return {"status": "error", "message": f"Selector query failed: {e}"}
       
        if not loc:
            try:
                n = await self.active_page.locator(selector).count()
            except Exception:
                n = 0
            if n == 0:
                return {"status": "error", "message" : f"No element for selector: {selector}"}
            
            return {
                "status": "error",
                "message": f"No visible and enabled element for selector: {selector}, ({n} node(s) matched but none were interactable)",
            }
 
        await loc.click(timeout=timeout_ms)
        await self._wait_for_load_state()
        return {"status": "success", "clicked_mode": "selector", "selector": selector, "url_after": self.active_page.url}

    async def _click_by_coordinates(
        self, coordinates: Optional[str]
    ) -> Dict[str, Any]:
        if not coordinates:
            return {"status": "error", "message": "mode='coordinates' requires: coordinates"}
        
        try:
            x, y = self._parse_point(coordinates)
            await self.active_page.mouse.click(x, y)
            clicked_text = await self.active_page.evaluate("""
                ({x, y}) => {
                    const el = document.elementFromPoint(x, y);
                    return el ? (el.innerText || el.textContent || "").slice(0, 100) : "";
                }
            """, {"x": x, "y": y})
        except Exception as e:
            logging.error(f"Coordinates click failed: {coordinates}, error: {e}")
            return {"status": "error", "message": f"Coordinates click failed: {e}"}
    
        await self._wait_for_load_state()
        return {"status": "success", "clicked_mode": "coordinates", "clicked_at": [x, y], "text": clicked_text, "url_after": self.active_page.url}

    async def click(
        self,
        mode: Literal["text", "selector", "coordinates"],
        text: Optional[str] = None,
        exact: bool = True,
        selector: Optional[str] = None,
        coordinates: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Click tool (multi-mode).

        Use cases:
        - mode="text": click using `text` as accessible name (button/link/textbox/searchbox/combobox),
          then label/aria-label, then visible substring text.
        Provide: text
        - mode="selector": click an element by CSS/XPath selector.
        Provide: selector
        - mode="coordinates": click at viewport coordinates "<point>x y</point>".
        Provide: coordinates

        Returns:
        dict with clicked info and url_after, or {"error": "..."}.
        """
        timeout_ms = 10000

        await self._ensure_started()
        async with self._page_lock:
            await self._wait_for_load_state()
            self._new_pages_opened = 0  # Reset counter before action
            try:
                result = None
                if mode == "text":
                    result = await self._click_by_text(text, exact=exact, timeout_ms=timeout_ms)

                elif mode == "selector":
                    result = await self._click_by_selector(selector, timeout_ms=timeout_ms)

                elif mode == "coordinates":
                    result = await self._click_by_coordinates(coordinates)                        

                else:
                    result = {"status": "error", "message": f"Unknown mode: {mode}"}

                # Add popup info if any new pages opened
                if self._new_pages_opened > 0:
                    result["new_pages_opened"] = self._new_pages_opened

                return result

            except PlaywrightTimeoutError as e:
                result = {"status": "error", "message": "Timeout during click", "mode": mode, "details": str(e)}
                if self._new_pages_opened > 0:
                    result["new_pages_opened"] = self._new_pages_opened
                return result

    async def switch_page(
        self,
        index: Optional[int] = None,
        url: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Switch the active page by index, URL substring, or title substring."""
        await self._ensure_started()
        async with self._page_lock:
            pages = self.context.pages
            if not pages:
                return {"status": "error", "message": "No pages available to switch."}

            selected_page = None
            if index is not None:
                if index < 0 or index >= len(pages):
                    return {"status": "error", "message": f"Page index out of range: {index}"}
                selected_page = pages[index]
            elif url:
                lower_url = url.lower()
                for page in pages:
                    if lower_url in page.url.lower():
                        selected_page = page
                        break
                if selected_page is None:
                    return {"status": "error", "message": f"No page found matching URL: {url}"}
            elif title:
                needle = title.lower()
                for page in pages:
                    try:
                        page_title = await page.title()
                    except Exception:
                        page_title = ""
                    if needle in page_title.lower():
                        selected_page = page
                        break
                if selected_page is None:
                    return {"status": "error", "message": f"No page found matching title: {title}"}
            else:
                return {"status": "error", "message": "Must provide index, url, or title to switch_page."}

            self.active_page = selected_page
            self.active_page.set_default_timeout(10000)
            return {
                "status": "success",
                "active_index": pages.index(selected_page),
                "url": selected_page.url,
                "title": await selected_page.title(),
            }

    async def type(self, selector: str, content: str):
        """Types into an input field safely."""
        await self._ensure_started()
        async with self._page_lock:
            logging.info(f"Typing into selector: {selector} with content: {content}")

            try:
                loc = await self._resolve_locator(selector, require_enabled=True, require_editable=True)
                if loc:
                    # Non blocking scroll to element
                    await self._scroll_into_view(loc, selector)
                    await loc.focus()
                    await loc.fill("")
                    await loc.type(content)
                    await self._wait_for_load_state()
                if not loc:
                    try:
                        n = await self.active_page.locator(selector).count()
                    except Exception:
                        n = 0
                    if n == 0:
                        return {"status": "error", "message": f"No element found for selector: {selector}"}
        
                    return {"status": "error", "message": f"No interactive element found for selector: {selector}, ({n} node(s) matched but none were interactable)"}
            except Exception as e:
                logging.warning(f"Selector query failed: {selector}, error: {e}")
                return {"status": "error", "message": f"Selector query failed: {e}"}

            return {"status": "success", "typed_into": selector, "content": content}

    async def _get_scroll_metrics(self) -> Dict[str, Any]:
        """
        Returns basic scroll metrics for the current page.
        """
        metrics = await self.active_page.evaluate("""
            () => {
                const scrollY = window.scrollY || window.pageYOffset;
                const viewportH = window.innerHeight;
                const viewportW = window.innerWidth;
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
                    viewportW,
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
        return {"status": "success"}

    async def scroll_y(self, y: int, before: Dict[str, Any]) -> Dict[str, Any]:
        if y is None:
            return {"status": "error", "message": "y required"}
        target = max(0, min(y, before["docH"] - before["viewportH"]))
        await self.active_page.evaluate(
            """({top}) => window.scrollTo({ top, left: 0 })""", {"top": target}
        )
        return {"status": "success"}

    async def scroll_to_selector(self, selector: str) -> Dict[str, Any]:
        if not selector:
            return {"status": "error", "message": "selector required"}
        try:
            loc = await self._resolve_locator(selector, require_enabled=False, require_editable=False)
        except Exception as e:
            logging.warning(f"Selector query failed: {selector}, error: {e}")
            return {"status": "error", "message": f"Selector query failed: {e}"}

        if not loc:
            try:
                n = await self.active_page.locator(selector).count()
            except Exception:
                n = 0
            if n == 0:
                return {"status": "error", "message" : f"No element for selector: {selector}"}
            
            return {
                "status": "error",
                "message": f"No visible element for selector: {selector}, ({n} node(s) matched but none were visible)",
            }
        await self._scroll_into_view(loc, selector)
        return {"status": "success"}

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
        return {"status": "success"}

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
        return {"status": "success"}

    async def scroll(
        self,
        mode: Literal["step", "percent", "y", "to_text", "to_selector"],
        direction: Literal["down", "up", "left", "right"],
        steps: int = 1,
        percent: Optional[float] = None,
        y: Optional[int] = None,
        text: Optional[str] = None,
        selector: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Multi-mode scroll tool.

        Modes:
        - step: scroll in small steps.
                Uses direction + steps.
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

            try:
                # get basic metrics
                before = await self._get_scroll_metrics()

                if mode == "percent":
                    result = await self.scroll_percent(percent, before)
                    if result["status"] == "error":
                        return result

                elif mode == "y":
                    result = await self.scroll_y(y, before)
                    if result["status"] == "error":
                        return result

                elif mode == "to_selector":
                    result = await self.scroll_to_selector(selector)
                    if result["status"] == "error":
                        return result

                elif mode == "to_text":
                    result = await self.scroll_to_text(text)
                    if result["status"] == "error":
                        return result

                else:  # step
                    result = await self.scroll_step(direction, steps, before)
                    if result["status"] == "error":
                        return result

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
            except Exception as e:
                logging.error(f"Scroll failed: {mode}, error: {e}")
                return {"status": "error", "message": f"Scroll failed: {e}"}

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
                e.g. ["Control", "A"] for Control+A, or ["Control", "Shift", "A"] for Control+Shift+A.

        Returns:
            A dict describing what was pressed.
        """
        await self._ensure_started()
        async with self._page_lock:
            await self._wait_for_load_state()

            try:
                def is_modifier(k: str) -> bool:
                    return k in {"Control", "Shift", "Alt", "Meta", "Command"}

                if len(keys) == 1:
                    await self.active_page.keyboard.press(keys[0])
                elif len(keys) > 1 and all(is_modifier(k) for k in keys[:-1]) and not is_modifier(keys[-1]):
                    # Handle combination like Control+A or Control+Shift+A
                    combo = "+".join(keys)
                    await self.active_page.keyboard.press(combo)
                else:
                    # Press sequentially
                    for key in keys:
                        await self.active_page.keyboard.press(key)

                await self._wait_for_load_state()

                return {"status": "success", "pressed_keys": keys, "url_after": self.active_page.url}
            except Exception as e:
                return {"status": "error", "message": f"Keyboard press failed: {str(e)}", "pressed_keys": keys}

    async def select_option(
        self,
        selector: str,
        value: Optional[str] = None,
        label: Optional[str] = None,
        index: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Selects an option inside a native <select> dropdown element.
        Use this tool instead of 'type' when the target is a <select> — 'type'/'fill' do not work on native dropdowns.

        selector: CSS selector targeting the <select> element.
        Provide exactly one of:
          - value: the option's HTML value attribute (e.g. value="IT")
          - label: the option's visible text as shown in the dropdown (e.g. "Italy")
          - index: 0-based position of the option in the list

        Internally calls Playwright's select_option(), which dispatches the change event
        so the page reacts exactly as if the user had picked the option manually.
        """
        await self._ensure_started()
        async with self._page_lock:
            if not any([value is not None, label is not None, index is not None]):
                return {"status": "error", "message": "Provide at least one of: value, label, or index"}
            try:
                loc = await self._resolve_locator(selector, require_enabled=True, require_editable=False)
                if not loc:
                    try:
                        n = await self.active_page.locator(selector).count()
                    except Exception:
                        n = 0
                    if n == 0:
                        return {"status": "error", "message": f"No element found for selector: {selector}"}
                    return {"status": "error", "message": f"No visible/enabled element for selector: {selector}"}

                if value is not None:
                    selected = await loc.select_option(value=value)
                elif label is not None:
                    selected = await loc.select_option(label=label)
                else:
                    selected = await loc.select_option(index=index)

                await self._wait_for_load_state()
                return {"status": "success", "selector": selector, "selected_values": selected}
            except Exception as e:
                return {"status": "error", "message": f"select_option failed: {str(e)}"}

    async def hover(
        self,
        mode: Literal["selector", "coordinates"],
        selector: Optional[str] = None,
        coordinates: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Moves the mouse over an element to trigger hover-based UI effects such as dropdown menus,
        tooltips, or any content that only appears on mouseenter/mouseover.
        Call this before clicking when the target element is only visible after a hover.

        mode="selector": hover over the element matching the CSS selector.
        mode="coordinates": hover at the given viewport coordinates "<point>x y</point>".

        Internally uses Playwright's locator.hover() (which also scrolls the element into view)
        or page.mouse.move() for coordinate-based hovering. Waits 500ms after the hover
        to let hover-triggered animations or menus render before the next action.
        """
        await self._ensure_started()
        async with self._page_lock:
            try:
                if mode == "selector":
                    if not selector:
                        return {"status": "error", "message": "mode='selector' requires: selector"}
                    loc = await self._resolve_locator(selector, require_enabled=False, require_editable=False)
                    if not loc:
                        return {"status": "error", "message": f"No visible element for selector: {selector}"}
                    await loc.hover()

                elif mode == "coordinates":
                    if not coordinates:
                        return {"status": "error", "message": "mode='coordinates' requires: coordinates"}
                    x, y = self._parse_point(coordinates)
                    await self.active_page.mouse.move(x, y)

                else:
                    return {"status": "error", "message": f"Unknown mode: {mode}"}

                # Wait for hover-triggered UI (menus, tooltips) to appear
                await self.active_page.wait_for_timeout(500)
                return {"status": "success", "mode": mode}
            except Exception as e:
                return {"status": "error", "message": f"hover failed: {str(e)}"}

    async def extract_content(
        self,
        selector: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extracts visible text content from the current page without relying on the screenshot.
        Use this when the task requires reading non-interactive content such as prices, search results,
        error messages, article text, or table data — elements that do not appear in the interactive DOM
        returned by get_state.

        selector: optional CSS selector to narrow extraction to a specific area of the page
                  (e.g. 'table', '.price', '#search-results', 'article').
                  If omitted, returns the full visible body text.

        When a selector is provided, uses Playwright's locator API so that extended selectors
        like :has-text(), :visible, and other Playwright pseudo-classes are supported alongside
        standard CSS. Falls back to document.body.innerText for full-page extraction.
        Output is truncated to 5000 chars to avoid excessive token usage.
        """
        await self._ensure_started()
        async with self._page_lock:
            try:
                if selector:
                    # Use Playwright's locator so extended selectors like :has-text() work.
                    # querySelectorAll only understands standard CSS and would throw on them.
                    loc = self.active_page.locator(selector)
                    count = await loc.count()
                    if count == 0:
                        return {"status": "error", "message": f"No content found for selector: {selector}"}
                    parts = []
                    for i in range(count):
                        try:
                            t = await loc.nth(i).inner_text()
                            if t.strip():
                                parts.append(t.strip())
                        except Exception:
                            pass
                    text = "\n---\n".join(parts)
                    if not text:
                        return {"status": "error", "message": f"No content found for selector: {selector}"}
                else:
                    text = await self.active_page.evaluate(
                        "() => (document.body.innerText || '').trim()"
                    )

                if len(text) > 5000:
                    text = text[:5000] + "\n...[truncated]"

                return {"status": "success", "content": text, "url": self.active_page.url}
            except Exception as e:
                return {"status": "error", "message": f"extract_content failed: {str(e)}"}

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
