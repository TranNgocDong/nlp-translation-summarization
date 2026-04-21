# NER configuration
NER_MODEL = "underthesea"  # Pre-trained

# Entity types
ENTITY_TYPES = [
    "PERSON",      # Nhân vật
    "LOCATION",    # Địa điểm
    "OBJECT",      # Vật dụng/Bảo vật
    "EVENT"        # Sự kiện
]

# Settings
CONFIDENCE_THRESHOLD = 0.5
ENABLE_CACHING = True
CACHE_SIZE = 1000