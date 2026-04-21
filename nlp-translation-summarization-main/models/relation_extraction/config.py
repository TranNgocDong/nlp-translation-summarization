# Relation types
RELATION_TYPES = {
    "DEFEATS": "A đánh bại B",
    "OBTAINS": "A lấy được / sở hữu B",
    "HELPS": "A giúp đỡ B",
    "HATES": "A ghét B",
    "LOVES": "A yêu B",
    "MEETS": "A gặp B",
    "PROTECTS": "A bảo vệ B",
    "NO_RELATION": "Không có quan hệ"
}

# Settings
CONFIDENCE_THRESHOLD = 0.7
MAX_DISTANCE_TOKENS = 15  # Khoảng cách tối đa giữa 2 entities

# Patterns để detect relations
RELATION_PATTERNS = {
    "DEFEATS": ["đánh bại", "chiến thắng", "hạ gục", "đánh", "chiến thắng"],
    "OBTAINS": ["lấy", "sở hữu", "kiếm được", "nhận", "chiếm"],
    "HELPS": ["giúp", "hỗ trợ", "cứu", "giải cứu"],
    "MEETS": ["gặp", "gặp mặt", "đến gặp", "gặp với"],
    "LOVES": ["yêu", "thương", "quý"],
    "HATES": ["ghét", "thù", "oán"],
    "PROTECTS": ["bảo vệ", "che chở", "bảo bọc"]
}