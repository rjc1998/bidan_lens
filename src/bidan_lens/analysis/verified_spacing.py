from __future__ import annotations

from types import MappingProxyType

# These mappings are deliberately exact and small. Additions require an authoritative
# source and focused false-promotion coverage; Kiwi output alone is not evidence.
VERIFIED_SPACING = MappingProxyType(
    {
        '갔다오다': '갔다 오다',
    }
)

VERIFIED_SPACING_SOURCES = MappingProxyType(
    {
        '갔다오다': (
            'National Institute of Korean Language, Online Q&A 300913',
            'https://www.korean.go.kr/front/onlineQna/onlineQnaView.do?mn_id=216&qna_seq=300913',
        ),
    }
)
