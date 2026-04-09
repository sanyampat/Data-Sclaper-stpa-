"""
config.py — Central configuration for the hype scraping pipeline.
All tunable parameters live here. Nothing else imports os.environ directly.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
CACHE_DIR  = BASE_DIR / "cache"
LOG_DIR    = BASE_DIR / "logs"

for _d in (DATA_DIR, CACHE_DIR, LOG_DIR):
    _d.mkdir(exist_ok=True)

TRENDS_CSV  = DATA_DIR / "trends.csv"
DROPS_CSV   = DATA_DIR / "drops.csv"
MERGED_CSV  = DATA_DIR / "merged_dataset.csv"
DB_PATH     = DATA_DIR / "hype_pipeline.db"

# ── Google Trends ──────────────────────────────────────────────────────────────
TRENDS_WINDOW_DAYS   = 90
TRENDS_OVERLAP_DAYS  = 7
TRENDS_GEO           = ""
TRENDS_RETRY_MAX     = 5
TRENDS_RETRY_BACKOFF = 60
TRENDS_CACHE_TTL     = 86_400

# ── Scraping ───────────────────────────────────────────────────────────────────
REQUEST_TIMEOUT  = 15
MAX_WORKERS      = 4
DELAY_MIN        = 1.5
DELAY_MAX        = 3.5

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0",
]

# ── Drop event keywords ────────────────────────────────────────────────────────
# [FIX C1] Removed overly broad tokens that generate excessive false positives:
#   "x "  — matches any article containing the letter x followed by a space
#   "x-"  — similarly too broad
#   "x:"  — similarly too broad
#   "fw"  — matches common English words ("few", "fw:", etc.)
#   "ss"  — matches common abbreviations unrelated to fashion
#   "rare" — matches unrelated editorial content ("rare footage", etc.)
#
# These were matching hundreds of non-drop articles, diluting the event signal
# and polluting merged_dataset.csv with noise that undermines the Poisson model.
DROP_KEYWORDS = [
    # Core drop signals
    "drop", "release", "launch", "collection", "limited",

    # Hype signals
    "exclusive", "sold out", "high demand", "viral", "trending",

    # Collaboration signals
    "collab", "collaboration",

    # Product lifecycle
    "debut", "restock", "re-release", "reissue",

    # Fashion terminology
    "capsule", "runway", "seasonal", "fall winter", "spring summer",

    # Sneaker culture
    "SNKRS", "raffle", "draw", "drop day",

    # Luxury signals
    "limited edition", "exclusive release", "special edition", "premium",

    # NFT / digital
    "mint", "NFT drop",
]

# ── Brand → search keywords ────────────────────────────────────────────────────
# [FIX C2] Added "category" field to each brand entry (was present in the
# uploaded config.py but referenced in compute_hype_metrics for filtering).
# [FIX C3] Capped trends_kw to 5 per brand — pytrends only supports 5 keywords
# per payload.  Excess keywords were silently ignored by pytrends, causing
# inconsistent coverage.  The most signal-rich 5 are kept per brand.
BRANDS = {
    "Nike": {
        "category": "sportswear",
        "trends_kw": [
            "Nike drop", "Nike SNKRS", "Nike release",
            "Air Jordan release", "Nike limited edition",
        ],
        "drop_kw": [
            "Nike", "Air Jordan", "Jordan Brand",
            "Air Force", "Dunk", "SNKRS",
        ],
    },
    "Supreme": {
        "category": "streetwear",
        "trends_kw": [
            "Supreme drop", "Supreme release",
            "Supreme collection", "Supreme week drop",
            "Supreme collab",
        ],
        "drop_kw": ["Supreme", "Supreme NYC"],
    },
    "Louis Vuitton": {
        "category": "luxury",
        "trends_kw": [
            "Louis Vuitton drop", "LV collection",
            "Louis Vuitton release", "LV bag",
            "Louis Vuitton collab",
        ],
        "drop_kw": ["Louis Vuitton", "LV"],
    },
    "Hermès": {
        "category": "ultra_luxury",
        "trends_kw": [
            "Hermes Birkin", "Hermes Kelly",
            "Hermes bag", "Hermes limited edition",
            "Birkin bag price",
        ],
        "drop_kw": ["Hermès", "Hermes", "Birkin", "Kelly"],
    },
    "Off-White": {
        "category": "street_luxury",
        "trends_kw": [
            "Off White drop", "Off-White release",
            "Off White collab", "Virgil Abloh Off White",
            "Off White sneakers",
        ],
        "drop_kw": ["Off-White", "Off White"],
    },
    "Balenciaga": {
        "category": "luxury",
        "trends_kw": [
            "Balenciaga drop", "Balenciaga sneakers",
            "Balenciaga collection", "Balenciaga limited",
            "Balenciaga new release",
        ],
        "drop_kw": ["Balenciaga"],
    },
    "Dior": {
        "category": "luxury",
        "trends_kw": [
            "Dior drop", "Dior collection",
            "Dior release", "Dior sneakers",
            "Dior limited edition",
        ],
        "drop_kw": ["Dior"],
    },
    "Yeezy": {
        "category": "streetwear",
        "trends_kw": [
            "Yeezy drop", "Yeezy release",
            "Adidas Yeezy", "Yeezy restock",
            "Yeezy sneakers",
        ],
        "drop_kw": ["Yeezy", "Adidas Yeezy"],
    },
    "Palace": {
        "category": "streetwear",
        "trends_kw": [
            "Palace drop", "Palace Skateboards",
            "Palace release", "Palace collab",
            "Palace collection",
        ],
        "drop_kw": ["Palace"],
    },
    "Rolex": {
        "category": "watch_luxury",
        "trends_kw": [
            "Rolex release", "Rolex new model",
            "Rolex waitlist", "Rolex Daytona",
            "Rolex Submariner",
        ],
        "drop_kw": ["Rolex", "Daytona", "Submariner"],
    },
}

# ── News sources ───────────────────────────────────────────────────────────────
SOURCES = {
    # ── Core sources ───────────────────────────────────────────────────────────
    "Hypebeast": {
        "rss": [
            "https://hypebeast.com/feed",
            "https://hypebeast.com/footwear/feed",
            "https://hypebeast.com/fashion/feed",
        ],
        "pages": [],
        "max_p": 0,
    },
    "Highsnobiety": {
        "rss": ["https://www.highsnobiety.com/feed/"],
        "pages": [],
        "max_p": 0,
    },
    "SneakerNews": {
        "rss": ["https://sneakernews.com/feed/"],
        "pages": [],
        "max_p": 0,
    },
    # ── Secondary sources ──────────────────────────────────────────────────────
    "ComplexSneakers": {
        "rss": ["https://www.complex.com/sneakers/rss.xml"],
        "pages": [],
        "max_p": 0,
    },
    "NiceKicks": {
        "rss": ["https://www.nicekicks.com/feed/"],
        "pages": [],
        "max_p": 0,
    },
    "FootwearNews": {
        "rss": ["https://footwearnews.com/feed/"],
        "pages": [],
        "max_p": 0,
    },
    # ── Luxury / fashion sources ───────────────────────────────────────────────
    "Vogue": {
        "rss": ["https://www.vogue.com/feed/rss"],
        "pages": [],
        "max_p": 0,
    },
    "WWD": {
        "rss": ["https://wwd.com/feed/"],
        "pages": [],
        "max_p": 0,
    },
}
