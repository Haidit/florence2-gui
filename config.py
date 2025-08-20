LOCAL_MODELS = {
    "Florence-2-base": {
        "path": "./models/florence-2-base",
        "description": "Base version of Florence-2"
    },
    "Florence-2-large": {
        "path": "./models/florence-2-large",
        "description": "Large version of Florence-2"
    }
}

VISUAL_TASKS = {
    "<OD>": "Object Detection",
    "<DENSE_REGION_CAPTION>": "Dense Region Caption",
    "<REGION_PROPOSAL>": "Region Proposal",
    "<CAPTION_TO_PHRASE_GROUNDING>": "Caption To Phrase Grounding",
    "<REFERRING_EXPRESSION_SEGMENTATION>": "Referring Expression Segmentation",
    "<REGION_TO_SEGMENTATION>": "Region to Segmentation",
    "<OPEN_VOCABULARY_DETECTION>": "Open Vocabulary Detection",
    "<OCR_WITH_REGION>": "OCR With Region"
}

TEXT_ONLY_TASKS = {
    "<OCR>": "OCR",
    "<CAPTION>": "Caption",
    "<DETAILED_CAPTION>": "Detailed Caption",
    "<REGION_TO_CATEGORY>": "Region To Category",
    "<REGION_TO_DESCRIPTION>": "Region To Description",
    "<REGION_TO_OCR>": "Region to OCR",
    "<MORE_DETAILED_CAPTION>": "More Detailed Caption",
}

TASK_TAGS = {
    "Object Detection": "<OD>",
    "Dense Region Caption": "<DENSE_REGION_CAPTION>",
    "Region Proposal": "<REGION_PROPOSAL>",
    "Caption To Phrase Grounding": "<CAPTION_TO_PHRASE_GROUNDING>",
    "Reffering Expression Segmentation": "<REFERRING_EXPRESSION_SEGMENTATION>",
    "Region to Segmentation": "<REGION_TO_SEGMENTATION>",
    "Open Vocabulary Detection": "<OPEN_VOCABULARY_DETECTION>",
    "Region To Category": "<REGION_TO_CATEGORY>",
    "Region To Description": "<REGION_TO_DESCRIPTION>",
    "Region to OCR": "<REGION_TO_OCR>",
    "OCR": "<OCR>",
    "OCR With Region": "<OCR_WITH_REGION>",
    "Caption": "<CAPTION>",
    "Detailed Caption": "<DETAILED_CAPTION>",
    "More Detailed Caption": "<MORE_DETAILED_CAPTION>"
    }