"""Curated conversation topics and seeded, explicit fact rendering."""

from dataclasses import dataclass
from datetime import date, timedelta
import random


_PERSON_FIRST_NAMES = (
    "Amira", "Ben", "Carmen", "Dev", "Elena", "Farah", "Gabriel", "Hana",
    "Isabel", "Jonah", "Keiko", "Luca", "Maya", "Noah", "Priya", "Samira",
)
_PERSON_LAST_NAMES = (
    "Ahmed", "Brooks", "Chen", "Diaz", "Evans", "Fernandez", "Gupta", "Hassan",
    "Ito", "Johnson", "Kim", "Lopez", "Morgan", "Patel", "Rivera", "Singh",
)
_PLACES = (
    "the Cedar Street library", "the Riverside community center",
    "the Maple Avenue cafe", "the north entrance of Green Park",
    "the Harbor View meeting room", "the Elm Street coworking studio",
    "the Oak Lane recreation hall", "the town hall courtyard",
    "the Lakeside visitor center", "the Pine Street art gallery",
    "the West End neighborhood center", "the Botanical Garden pavilion",
    "the railway station's main lobby", "the Hillcrest sports clubhouse",
    "the Market Square information desk", "the university's east auditorium",
)
_PREFERRED_TIMES = (
    "early morning", "late morning", "around noon", "early afternoon",
    "late afternoon", "early evening", "late evening",
)
_DECISIONS = (
    "proceed with the original plan", "run a small pilot first",
    "postpone until next month", "ask for more proposals before proceeding",
    "reduce the scope before proceeding", "cancel the plan",
    "seek a partner before proceeding", "request community feedback first",
)


@dataclass(frozen=True)
class Topic:
    """A named fact slot, with a natural value domain and reusable wording."""

    topic_id: str
    label: str
    category: str
    slot_name: str
    slot_type: str
    introduction_templates: tuple[str, ...]
    reference_templates: tuple[str, ...]

    def value(self, rng: random.Random) -> str:
        """Draw a fact value using only the caller's random state."""
        if self.slot_type == "person":
            return f"{rng.choice(_PERSON_FIRST_NAMES)} {rng.choice(_PERSON_LAST_NAMES)}"
        if self.slot_type == "place":
            return rng.choice(_PLACES)
        if self.slot_type == "time":
            return rng.choice(_PREFERRED_TIMES)
        if self.slot_type == "date":
            return (date(2027, 1, 1) + timedelta(days=rng.randrange(1096))).isoformat()
        if self.slot_type == "decision":
            return rng.choice(_DECISIONS)
        if self.slot_type == "money":
            return f"${rng.randrange(10, 10001) * 5:,}"
        raise ValueError(f"Unsupported topic slot type: {self.slot_type}")


# Each group shares a coherent fact slot, but names distinct real-life concerns.
_TOPIC_GROUPS = (
    (
        "people", "designated contact", "person",
        (
            "family reunion", "school fundraiser", "neighborhood watch",
            "book club", "youth soccer team", "apartment maintenance",
            "garden volunteers", "language exchange", "charity auction",
            "wedding planning", "choir rehearsals", "museum membership",
            "cycling club", "animal shelter volunteering", "parent council",
            "workplace mentoring", "community theater", "local history society",
        ),
        (
            'For {label}, my {slot_name} is "{value}".',
            'Please remember "{value}" as the {slot_name} for {label}.',
            'I have assigned "{value}" as the {slot_name} for {label}.',
        ),
        (
            'I am following up on {label}; keep the {slot_name} in mind.',
            'For {label}, please contact the {slot_name} we discussed earlier.',
        ),
    ),
    (
        "places", "meeting location", "place",
        (
            "weekly study group", "photography walk", "pottery workshop",
            "chess tournament", "resume coaching", "poetry reading",
            "newcomer orientation", "citizen science briefing", "repair cafe",
            "walking group", "oral history interview", "cooking class",
            "first aid course", "knitting circle", "board game night",
            "birdwatching outing", "public speaking practice", "recycling drive",
        ),
        (
            'The {slot_name} for {label} is "{value}".',
            'For {label}, we will meet at "{value}"; that is the {slot_name}.',
            'Please record "{value}" as the {slot_name} for {label}.',
        ),
        (
            'For {label}, let us use the {slot_name} we previously arranged.',
            'I am preparing for {label}; please remember the {slot_name}.',
        ),
    ),
    (
        "preferences", "preferred time of day", "time",
        (
            "reading novels", "going for a run", "practicing piano",
            "shopping for groceries", "writing in my journal", "doing yoga",
            "calling relatives", "working on puzzles", "watering houseplants",
            "baking bread", "studying vocabulary", "taking photographs",
            "planning meals", "listening to podcasts", "meditating",
            "cleaning the kitchen", "walking the dog", "sketching outdoors",
        ),
        (
            'For {label}, my {slot_name} is "{value}".',
            'I prefer "{value}" for {label}; please remember that {slot_name}.',
            'My {slot_name} for {label} is currently "{value}".',
        ),
        (
            'When planning {label}, please keep my {slot_name} in mind.',
            'For {label}, use the {slot_name} I told you earlier.',
        ),
    ),
    (
        "events", "scheduled date", "date",
        (
            "annual dental checkup", "cousin's graduation party",
            "roof inspection", "passport appointment", "car service appointment",
            "spring concert", "summer picnic", "autumn craft fair",
            "winter charity dinner", "employee retreat", "alumni reunion",
            "home energy assessment", "vaccination appointment",
            "library anniversary celebration", "river cleanup day",
            "science festival", "sports awards ceremony", "business opening",
        ),
        (
            'The {slot_name} for {label} is "{value}".',
            'I have put {label} on the calendar with a {slot_name} of "{value}".',
            'Please remember "{value}" as the {slot_name} for {label}.',
        ),
        (
            'Please plan around the {slot_name} we discussed for {label}.',
            'I am preparing for {label}; keep its {slot_name} in mind.',
        ),
    ),
    (
        "decisions", "chosen course of action", "decision",
        (
            "kitchen renovation", "website redesign", "office relocation",
            "solar panel installation", "community composting proposal",
            "school playground upgrade", "new delivery service",
            "club membership expansion", "product launch", "archival digitization",
            "rainwater collection project", "shared workshop proposal",
            "accessible entrance project", "mentoring program expansion",
            "neighborhood newsletter", "equipment replacement",
            "weekend market proposal", "public mural project",
        ),
        (
            'For {label}, our {slot_name} is "{value}".',
            'We have decided to "{value}" for {label}; that is our {slot_name}.',
            'Please record "{value}" as the {slot_name} for {label}.',
        ),
        (
            'For {label}, let us follow the {slot_name} we discussed earlier.',
            'Please keep our {slot_name} in mind when planning {label}.',
        ),
    ),
    (
        "quantities", "allocated budget in US dollars", "money",
        (
            "camping holiday", "birthday gifts", "computer purchase",
            "emergency savings", "furniture purchase", "bicycle repairs",
            "professional training", "garden supplies", "school supplies",
            "pet care", "home security upgrade", "music lessons",
            "conference travel", "sports equipment", "holiday decorations",
            "charitable donations", "moving expenses", "camera equipment",
        ),
        (
            'For {label}, my {slot_name} is "{value}".',
            'I have set the {slot_name} for {label} to "{value}".',
            'Please remember "{value}" as the {slot_name} for {label}.',
        ),
        (
            'For {label}, please work within the {slot_name} we discussed.',
            'I am planning {label}; keep its {slot_name} in mind.',
        ),
    ),
)


class TopicPool:
    """The stable ordered pool of 108 distinct conversation topics."""

    def __init__(self) -> None:
        self.topics = tuple(
            Topic(
                topic_id=f"{category}-{index:02d}",
                label=label,
                category=category,
                slot_name=slot_name,
                slot_type=slot_type,
                introduction_templates=introductions,
                reference_templates=references,
            )
            for category, slot_name, slot_type, labels, introductions, references
            in _TOPIC_GROUPS
            for index, label in enumerate(labels, start=1)
        )
        self.by_id = {topic.topic_id: topic for topic in self.topics}


class TurnRenderer:
    """Render introductions, updates, references and explicit expiry notices."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def introduce(
        self, topic: Topic, value: str, previous_value: str | None = None
    ) -> str:
        if previous_value is not None:
            template = self.rng.choice((
                'Update for {label}: the {slot_name} was "{previous_value}"; '
                'it is now "{value}". The previous value is no longer current.',
                'For {label}, replace the previous {slot_name} '
                '"{previous_value}" with "{value}". Only the new value is current.',
            ))
            return template.format(
                label=topic.label, slot_name=topic.slot_name,
                previous_value=previous_value, value=value,
            )
        return self.rng.choice(topic.introduction_templates).format(
            label=topic.label, slot_name=topic.slot_name, value=value,
        )

    def reference(self, topic: Topic, value: str) -> str:
        return self.rng.choice(topic.reference_templates).format(
            label=topic.label, slot_name=topic.slot_name, value=value,
        )

    def expire(self, topic: Topic, value: str) -> str:
        return self.rng.choice((
            'For {label}, the {slot_name} "{value}" has expired. '
            'There is no current value for this slot.',
            'Remove "{value}" as the current {slot_name} for {label}. '
            'It is no longer valid, and there is no current replacement value.',
        )).format(label=topic.label, slot_name=topic.slot_name, value=value)
