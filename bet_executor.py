"""
Automated bet execution for melbet-et.com (ETB account).

Setup:
    pip install playwright
    playwright install chromium

Usage:
    # Step 1 — save your login session (run once, or when session expires):
    python bet_executor.py auth

    # Step 2 — place a bet (uncomment final click when ready):
    python bet_executor.py bet <match_url> "<odds_selector>" <stake_etb>

Example:
    python bet_executor.py bet \
        "https://melbet-et.com/en/line/football/123456-team-a-vs-team-b" \
        "[data-outcome-id='987654']" \
        175
"""
import asyncio
import random
import sys
from pathlib import Path

from playwright.async_api import BrowserContext, async_playwright

AUTH_FILE  = "auth_state.json"
BASE_URL   = "https://melbet-et.com"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
LAUNCH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--no-first-run",
]
STEALTH_SCRIPT = """
    Object.defineProperty(navigator, 'webdriver',  { get: () => undefined });
    Object.defineProperty(navigator, 'plugins',    { get: () => [1, 2, 3, 4, 5] });
    Object.defineProperty(navigator, 'languages',  { get: () => ['en-US', 'en'] });
    Object.defineProperty(navigator, 'hardwareConcurrency', { get: () => 8 });
    window.chrome = { runtime: {}, loadTimes: () => {}, csi: () => {}, app: {} };
"""


async def _human_delay(lo: int = 300, hi: int = 900) -> None:
    await asyncio.sleep(random.uniform(lo, hi) / 1000)


async def _build_context(p, headless: bool = False, load_auth: bool = False) -> tuple:
    browser = await p.chromium.launch(
        headless=headless,
        args=LAUNCH_ARGS,
        slow_mo=40,
    )
    kwargs = dict(
        viewport={"width": 1366, "height": 768},
        user_agent=USER_AGENT,
        locale="en-US",
        timezone_id="Africa/Addis_Ababa",
    )
    if load_auth:
        kwargs["storage_state"] = AUTH_FILE
    context: BrowserContext = await browser.new_context(**kwargs)
    await context.add_init_script(STEALTH_SCRIPT)
    return browser, context


# ── 1. Authentication ────────────────────────────────────────────────────────

async def save_auth() -> None:
    """Open a real browser window for manual login, then persist the session."""
    async with async_playwright() as p:
        browser, context = await _build_context(p, headless=False)
        page = await context.new_page()
        await page.goto(BASE_URL, wait_until="domcontentloaded")

        print("\nA browser window has opened.")
        print("Log in to your melbet-et.com account normally.")
        print("Once you are fully logged in, come back here and press Enter.")
        input("  → Press Enter when logged in: ")

        await context.storage_state(path=AUTH_FILE)
        print(f"\nSession saved to '{AUTH_FILE}'. You won't need to log in again until it expires.\n")
        await browser.close()


# ── 2. Bet Execution ─────────────────────────────────────────────────────────

async def place_bet(match_url: str, odds_selector: str, stake_etb: float) -> None:
    """
    Navigate to a match, select an outcome, enter stake, and (optionally) confirm.

    odds_selector:
        The CSS selector for the specific odds button you want to click.
        Find it by right-clicking the odds on the site → Inspect → copy selector.
        Example: '[data-outcome-id="987654"]'

    stake_etb:
        Stake in Ethiopian Birr.
    """
    if not Path(AUTH_FILE).exists():
        raise FileNotFoundError(
            f"'{AUTH_FILE}' not found. Run 'python bet_executor.py auth' first."
        )

    async with async_playwright() as p:
        browser, context = await _build_context(p, headless=False, load_auth=True)
        page = await context.new_page()

        # ── Navigate ──────────────────────────────────────────────────────────
        print(f"\nNavigating to match...")
        await page.goto(match_url, wait_until="networkidle", timeout=30_000)
        await _human_delay(1000, 2000)

        # ── Click odds button ─────────────────────────────────────────────────
        print(f"Selecting odds: {odds_selector}")
        odds_btn = page.locator(odds_selector).first
        await odds_btn.wait_for(state="visible", timeout=15_000)
        await odds_btn.scroll_into_view_if_needed()
        await _human_delay(400, 800)
        await odds_btn.click()
        await _human_delay(700, 1200)

        # ── Enter stake ───────────────────────────────────────────────────────
        print(f"Entering stake: {stake_etb:.0f} ETB")
        stake_input = page.locator(
            "input.betslip__input, "
            "input[data-testid='stake-input'], "
            "input[name='stake'], "
            ".bet-slip input[type='number'], "
            ".betslip input[type='text']"
        ).first
        await stake_input.wait_for(state="visible", timeout=10_000)
        await stake_input.triple_click()
        await _human_delay(200, 400)
        await stake_input.type(str(int(stake_etb)), delay=90)
        await _human_delay(600, 1000)

        # ── Handle "Odds have changed" popup ──────────────────────────────────
        try:
            accept_btn = page.locator(
                "button:has-text('Accept'), "
                "button:has-text('Accept changes'), "
                "button:has-text('OK'), "
                "[data-testid='accept-odds-change']"
            ).first
            await accept_btn.wait_for(state="visible", timeout=4_000)
            print("  ⚠  Odds changed popup detected — accepting new odds.")
            await _human_delay(300, 500)
            await accept_btn.click()
            await _human_delay(500, 900)
        except Exception:
            pass  # No popup — continue normally

        # ── Locate place-bet button ───────────────────────────────────────────
        place_btn = page.locator(
            "button:has-text('Place bet'), "
            "button:has-text('Place Bet'), "
            "button:has-text('Confirm'), "
            "button[data-testid='place-bet-btn'], "
            ".betslip__place-btn"
        ).first
        await place_btn.wait_for(state="visible", timeout=10_000)
        btn_text = (await place_btn.text_content() or "").strip()

        print(f"\n{'─'*50}")
        print(f"  Match : {match_url}")
        print(f"  Odds  : {odds_selector}")
        print(f"  Stake : {stake_etb:.0f} ETB")
        print(f"  Button: '{btn_text}' — located and ready")
        print(f"{'─'*50}")
        print("\n  Final click is COMMENTED OUT for safety.")
        print("  Verify everything looks correct in the browser window.")
        print("  Then uncomment the line below and re-run.\n")

        # ✅ Remove the comment on the next line when you're ready to go live:
        # await place_btn.click()

        input("  → Press Enter to close the browser: ")
        await browser.close()


# ── CLI entry point ───────────────────────────────────────────────────────────

USAGE = """
Usage:
  python bet_executor.py auth
  python bet_executor.py bet <match_url> "<odds_selector>" <stake_etb>
"""

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "auth":
        asyncio.run(save_auth())

    elif cmd == "bet":
        if len(sys.argv) != 5:
            print(USAGE)
            sys.exit(1)
        _, _, url, selector, stake = sys.argv
        asyncio.run(place_bet(url, selector, float(stake)))

    else:
        print(f"Unknown command '{cmd}'.{USAGE}")
        sys.exit(1)
