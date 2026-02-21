import base64
import time
from playwright.async_api import async_playwright
from pydantic import BaseModel


class BrowserState(BaseModel):
    page_url: str
    page_screenshot_base64: str
    interactive_elements: list



class BrowserManager:
    def __init__(self, show_browser: bool = True):
        self.show_browser = show_browser
        self.playwright = None
        self.driver = None
        self.context = None
        self.active_page = None

    async def init(self):
        self.playwright = await async_playwright().start()
        self.driver = await self.playwright.chromium.launch(
            headless=not self.show_browser
        )

        self.context = await self.driver.new_context(
            record_video_dir="videos/",
            record_video_size={"width": 1280, "height": 720}
        )

        self.active_page = await self.context.new_page()


    async def _wait_for_load_state(self):
        """Waits for the page to be idle after an action."""
        try:
            await self.active_page.wait_for_load_state("networkidle", timeout=5000)
        except:
            pass


    def _parse_point(self, point: str):
        import re
        clean = point.replace("<point>", "").replace("</point>", "").strip()
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

    async def get_state(self) -> BrowserState:
        """
        Returns the full observable state of the browser.

        Includes:
        - Current page URL
        - Screenshot of the visible viewport
        - Structured list of interactive DOM elements
        (buttons, links, inputs, etc.) with bounding boxes

        Use this tool whenever you need to inspect the page
        before deciding the next action.
        """

        await self._wait_for_load_state()

        img_b64 = await self._take_screenshot()
        dom = await self._extract_interactive_elements()

        return {
            "page_url": self.active_page.url,
            "page_screenshot_base64": f"data:image/png;base64,{img_b64}",
            "interactive_elements": dom
        }
        
    async def goto_url(self, url: str):
        """Navigates to the specified URL.
        Args:
            url: The web address to visit (e.g., 'google.com' or 'https://openai.com').
        """
        if not url.startswith("http://") and not url.startswith("https://"):
            print(f"🔄 Aggiungo https:// a '{url}'...")
            url = "https://" + url

        try:
            # Aggiungiamo un timeout per evitare che si blocchi per sempre
            await self.active_page.goto(url, timeout=10000)
            return f"Navigated to {url}"
        except Exception as e:
            return f"Error navigating to {url}: {str(e)}"

    async def click_by_coordinates(self, coordinates: str):
        """Clicks at the specified coordinates.
        Use this only if you can't uniquely identify an element via DOM.
        Args:
            coordinates: A string in the format '<point>x y</point>' representing the point
        """
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
        Use this whenever you can uniquely identify an element via DOM.
        Example: 'input[type="text"]', '//div[@class="myClass"]', '//button[contains(text(), "Hello")]'
        """
        await self._wait_for_load_state()

        element = await self.active_page.query_selector(selector)
        if not element:
            return {"error": f"No element found for selector: {selector}"}

        await element.click()

        return {
            "clicked_selector": selector,
            "url_after": self.active_page.url
        }

    async def click_by_text(self, text: str, exact: bool = False):
        """
        Click the first visible element containing the given text.
        Prefer this over coordinates.
        """
        await self._wait_for_load_state()

        if exact:
            locator = self.active_page.locator(f"text=\"{text}\"")
        else:
            locator = self.active_page.locator(f"text={text}")

        count = await locator.count()

        if count == 0:
            return {"error": f"No element found containing text: {text}"}

        await locator.first.click()

        return {
            "clicked_text": text,
            "url_after": self.active_page.url
        }

    async def type(self, selector: str, content: str):
        """
        Types into an input field safely.
        - Scrolls into view
        - Focuses the input
        - Clears any existing text
        - Types the new content
        """
        element = await self.active_page.query_selector(selector)
        if not element:
            return {"error": f"No element found for selector: {selector}"}
        
        await element.scroll_into_view_if_needed()        
        await element.focus()    
        await element.fill("")  
        await element.type(content)
        await self._wait_for_load_state()
        return {"typed_into": selector, "content": content}


    async def scroll(self, point: str, direction: str):
        """Scrolls the page in the specified direction from the given coordinates.
        Args:
            point: A string in the format '<point>x y</point>' representing the coordinates to scroll from.
            direction: The direction to scroll ('up', 'down', 'left', 'right').
        """
        x, y = self._parse_point(point)
        await self.active_page.mouse.move(x, y)
        # ... resto della logica scroll ...
        if direction == "down":
            await self.active_page.mouse.wheel(0, 1000)
        elif direction == "up":
            await self.active_page.mouse.wheel(0, -1000)
        elif direction == "right":
            await self.active_page.mouse.wheel(1000, 0)
        elif direction == "left":
            await self.active_page.mouse.wheel(-1000, 0)
        await self._wait_for_load_state()
        return f"Scrolled {direction} at ({x}, {y})"

    async def wait(self):
        """Waits for a short period to allow the page to update."""
        await self.active_page.wait_for_timeout(5000)
        return "Waited for 5 seconds"

    async def close(self):
        """Closes the browser and cleans up resources."""
        if self.context:
            await self.context.close()
        if self.driver:
            await self.driver.close()
        if self.playwright:
            await self.playwright.stop()
