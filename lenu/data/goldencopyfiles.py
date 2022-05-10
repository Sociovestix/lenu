import json
import logging
from datetime import datetime
from typing import Optional, List

import requests
from pydantic import BaseModel, Field, ValidationError

GLEIF_GOLDEN_COPY_URL = (
    "https://leidata-preview.gleif.org/api/v2/golden-copies/publishes"
)

logger = logging.getLogger(__name__)


class FileRef(BaseModel):
    type: str
    format: str
    record_count: int
    size: int
    size_human_readable: str
    delta_type: str
    url: str


class FileInVariousFormats(BaseModel):
    csv: FileRef
    json_: FileRef = Field(alias="json")
    xml: FileRef


class DeltaFiles(BaseModel):
    IntraDay: Optional[FileInVariousFormats]
    LastDay: Optional[FileInVariousFormats]
    LastWeek: Optional[FileInVariousFormats]
    LastMonth: Optional[FileInVariousFormats]


class GoldenCopyFileBundle(BaseModel):
    type: str
    publish_date: datetime
    full_file: FileInVariousFormats
    delta_files: DeltaFiles  # can be [] as well


class GoldenCopyFilePublication(BaseModel):
    publish_date: datetime
    lei2: GoldenCopyFileBundle
    rr: GoldenCopyFileBundle
    repex: GoldenCopyFileBundle


class GoldenCopyFilePublications:
    """
    Allows for fetching information about GLEIF Golden Copy files.
    """

    def __init__(self, url: str = GLEIF_GOLDEN_COPY_URL, page_size=10):
        self._url = url
        self._page_size = page_size

    def _fetch_page(self, page: int) -> List[GoldenCopyFilePublication]:
        logger.debug('Fetch list of golden copy files from "{0}"'.format(self._url))
        response = requests.get(
            self._url, params={"page": page, "page_size": self._page_size}
        )
        publications = response.json()["data"]

        result = []
        for p in publications:
            try:
                result.append(GoldenCopyFilePublication.parse_obj(p))
            except ValidationError:
                p_str = json.dumps(p, indent=2)
                logger.exception(f"could not parse \n {p_str}")
                raise
        return result

    def fetch_latest(self) -> GoldenCopyFilePublication:
        page_publications = self._fetch_page(page=1)
        return list(
            sorted(page_publications, key=lambda p: p.publish_date, reverse=True)
        )[0]
