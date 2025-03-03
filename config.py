"""
Cats Module – Configuration for Dashboard Visuals
===================================================

This module centralizes all configuration settings for the dashboard. It defines
settings for emotions, products, companies, and service aspects. Rather than using
loose dictionaries, we use frozen dataclasses to group related values together. This
approach improves readability, type-safety, and prevents accidental modification at runtime.

Usage:
    Import the configuration objects into your other modules as needed:

        from cats import EMOTION_CONFIG, PRODUCT_CONFIG, COMPANY_CONFIG, ASPECT_CONFIG

    Then access properties such as:
        - PRODUCT_CONFIG.colours  (for product color mappings)
        - PRODUCT_CONFIG.emoji_map  (for product emoji labels)
        - COMPANY_CONFIG.insurer_colours (for company color mappings)
        - ASPECT_CONFIG.aspects_map  (for mapping aspect names to display strings)

Benefits:
    - Centralized settings make it easy to update visual styles or mappings across the entire app.
    - Grouping related values into classes ensures that changes are consistent and reduces the risk
      of typos or misconfiguration.
    - The immutable (frozen) design means that these configurations can’t be modified at runtime,
      increasing reliability.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class EmotionConfig:
    """
    Configuration for emotion-related settings.

    Attributes:
        colours (Dict[str, str]): Maps emotion names to color codes.
        weights (Dict[str, float]): Numeric weight assigned to each emotion.
        categories (List[str]): List of emotion keys derived from the colours mapping.
    """
    colours: Dict[str, str] = field(default_factory=lambda: {
        'joy': '#008000',  # green
        'neutral': '#FFFF00',  # yellow
        'surprise': '#808080',  # grey
        'sadness': '#FFC0CB',  # pink
        'anger': '#FF0000',  # red
        'disgust': '#800000',  # maroon
        'fear': '#FFA500',  # orange
    })
    weights: Dict[str, float] = field(default_factory=lambda: {
        'joy': 1.5,
        'neutral': -0.5,
        'sadness': -1.0,
        'anger': -2.0,
        'disgust': -2.0,
        'fear': 0,
        'surprise': 0,
    })
    categories: List[str] = field(default_factory=lambda: [
        'joy', 'neutral', 'surprise', 'sadness', 'anger', 'disgust', 'fear'
    ])


@dataclass(frozen=True)
class ProductConfig:
    """
    Configuration for product-related settings.

    Attributes:
        colours (Dict[str, str]): Maps product categories to color codes.
        emoji_map (Dict[str, str]): Maps product categories to emoji-enhanced labels.
        categories (List[str]): List of product categories derived from the colours mapping.
    """
    colours: Dict[str, str] = field(default_factory=lambda: {
        'Energy': '#FFA500',  # orange
        'Appliance Repair': '#800000',  # maroon
        'Home Electrical': '#800000',
        'Gas Products': '#008000',  # green
        'Plumbing & Drains': '#FF00FF',  # magenta
        'Unknown': '#808080',  # grey
    })
    emoji_map: Dict[str, str] = field(default_factory=lambda: {
        'All': "🌎 All",
        'Gas Products': "🚿 Gas Products",
        'Energy': "⚡ Energy",
        'Plumbing & Drains': "🪠 Plumbing & Drains",
        'Appliance Cover': "📺 Appliance Cover",
        'Home Electrical': "🔦 Home Electrical",
        'Heating': "🔥 Heating",
        'Pest Control': "🐀 Pest Control",
        'Unknown': "🃏 Unknown",
    })
    categories: List[str] = field(default_factory=lambda: [
        'Energy', 'Appliance Repair', 'Home Electrical', 'Gas Products', 'Plumbing & Drains', 'Unknown'
    ])


@dataclass(frozen=True)
class CompanyConfig:
    """
    Configuration for company-related settings.

    Attributes:
        emoji_map (Dict[str, str]): Maps company names to emoji-enhanced labels.
        insurer_colours (Dict[str, str]): Maps company names to color codes for charting.
    """
    emoji_map: Dict[str, str] = field(default_factory=lambda: {
        "British Gas": "🌎 British Gas",
        "HomeServe": "🧮 HomeServe",
        "CheckATrade": "🧮 CheckATrade",
        "Domestic & General": "🧮 Domestic & General",
        "Corgi HomePlan": "🧮 Corgi HomePlan",
        "247 Home Rescue": "🧮 247 Home Rescue",
        "Octopus": "⚡ Octopus",
        "OVO": "⚡ OVO",
    })
    insurer_colours: Dict[str, str] = field(default_factory=lambda: {
        "British Gas": "#0000FF",  # blue
        "HomeServe": "#da3d34",  # dark red
        "Domestic & General": "#00a480",  # green
        "Corgi HomePlan": "#ed9f40",  # orange
        "247 Home Rescue": "#ff78cb",  # pink
        "CheckATrade": "#6a76ac",  # dark blue
    })



@dataclass(frozen=True)
class AspectConfig:
    """
    Configuration for service aspect settings.

    Attributes:
        aspects_map (Dict[str, str]): Maps internal aspect keys to display labels (with emoji).
        aspect_colours (Dict[str, str]): Maps each aspect to its designated color.
        aspects (List[str]): List of aspect keys.
    """
    aspects_map: Dict[str, str] = field(default_factory=lambda: {
        "Appointment Scheduling": "⌚ Appointment Scheduling",
        "Customer Service": "📞 Customer Service",
        "Response Speed": "🥇 Response Speed",
        "Engineer Experience": "🧑‍🔧 Engineer Experience",
        "Solution Quality": "🧠 Solution Quality",
        "Value For Money": "💵 Value For Money",
    })
    aspect_colours: Dict[str, str] = field(default_factory=lambda: {
        "Appointment Scheduling": "#6a76ac",  # dark blue
        "Customer Service": "#da3d34",  # dark red
        "Response Speed": "#ed9f40",  # orange
        "Engineer Experience": "#00c2e0",  # light blue
        "Solution Quality": "#ff78cb",  # pink
        "Value For Money": "#00a480",  # green
    })
    aspects: List[str] = field(default_factory=lambda: [
        "Appointment Scheduling",
        "Customer Service",
        "Response Speed",
        "Engineer Experience",
        "Solution Quality",
        "Value For Money",
    ])


# Instantiate configuration objects as module-level constants.
# These constants are intended to be imported by other modules (e.g., your charts module).
EMOTION_CONFIG = EmotionConfig()
PRODUCT_CONFIG = ProductConfig()
COMPANY_CONFIG = CompanyConfig()
ASPECT_CONFIG = AspectConfig()
