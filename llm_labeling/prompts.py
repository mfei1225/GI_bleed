"""Prompts used to label radiology reports for the GI-bleed cohort.

Two tasks:

- ``injury``  → Bowel / Kidney / Liver / Extravasation (Yes|No|Maybe)
- ``location`` → organ flags for reports already called positive
"""

INJURY_SYSTEM = (
    "You label finalized radiology reports from abdominopelvic CT angiography. "
    "Reply with a single JSON object and nothing else."
)

INJURY_USER = """Read the radiology report below.

Decide four labels. Use only Yes, No, or Maybe.

- Extravasation: is there active contrast extravasation or active GI bleeding?
  Yes = the report documents active extravasation / active GI bleed / active blush.
  No = the report explicitly denies active extravasation and active GI bleeding.
  Maybe = hedging, old blood without a current blush, or it is unclear.
- Bowel: is the bleed (or the finding you labeled) in the GI tract
  (esophagus, stomach, duodenum, small bowel, colon, rectum)?
- Kidney: kidney / collecting system involvement of the bleed or hematoma.
- Liver: liver involvement of the bleed or hematoma.

Do not treat extra-GI hematoma (spleen, psoas, abdominal wall) as Extravasation=Yes
unless the report also describes active contrast extravasation.

Return JSON:
{{
  "Extravasation": "Yes|No|Maybe",
  "Bowel": "Yes|No|Maybe",
  "Kidney": "Yes|No|Maybe",
  "Liver": "Yes|No|Maybe",
  "Explanation": "one short sentence"
}}

REPORT:
{report}
"""

LOCATION_SYSTEM = (
    "You assign anatomic sites of active bleeding from a radiology report. "
    "Reply with a single JSON object and nothing else."
)

LOCATION_USER = """This report was already judged to describe active extravasation or GI bleeding.
Mark each site Yes or No. Yes only if that site is a described location of
active bleed / extravasation / blush. Hematoma next to an organ without a blush
is No.

Sites:
- BOWELS (small bowel, colon, rectum, anastomosis — not stomach/esophagus)
- STOMACH
- ESOPHAGUS
- LIVER
- SPLEEN
- KIDNEY
- ABDOMINAL WALL
- GLUTEALS/THIGHS
- RETROPERITONEAL
- OTHER (any other named site)

Return JSON with these exact keys and nothing else. Values are Yes or No only:
{{
  "BOWELS": "Yes|No",
  "LIVER": "Yes|No",
  "SPLEEN": "Yes|No",
  "KIDNEY": "Yes|No",
  "ABDOMINAL WALL": "Yes|No",
  "GLUTEALS/THIGHS": "Yes|No",
  "ESOPHAGUS": "Yes|No",
  "STOMACH": "Yes|No",
  "RETROPERITONEAL": "Yes|No",
  "OTHER": "Yes|No"
}}

REPORT:
{report}
"""

LOCATION_KEYS = (
    "BOWELS",
    "LIVER",
    "SPLEEN",
    "KIDNEY",
    "ABDOMINAL WALL",
    "GLUTEALS/THIGHS",
    "ESOPHAGUS",
    "STOMACH",
    "RETROPERITONEAL",
    "OTHER",
)

INJURY_KEYS = ("Bowel", "Kidney", "Liver", "Extravasation")
ALLOWED = {"Yes", "No", "Maybe"}
