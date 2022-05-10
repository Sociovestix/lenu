import pandas  # type: ignore

# some columns names as constants for quick reuse
COL_LEGALNAME = "Entity.LegalName"
COL_LEGALNAME_LANG = "Entity.LegalName.xmllang"
COL_JURISDICTION = "Entity.LegalJurisdiction"
COL_ELF = "Entity.LegalForm.EntityLegalFormCode"


def load_lei_cdf_data(url, usecols=None):
    return pandas.read_csv(
        url,
        compression="zip",
        low_memory=False,
        dtype=str,
        # the following will prevent pandas from converting strings like 'NA' to NaN.
        na_values=[""],
        keep_default_na=False,
        usecols=usecols,
    )


def get_legal_jurisdiction(lei):
    if lei["Entity.LegalJurisdiction"] == "US":
        # this means we have a ISO-3166-2 code here
        if pandas.notnull(lei["Entity.LegalAddress.Region"]):
            return lei["Entity.LegalAddress.Region"]
        else:
            return lei["Entity.LegalJurisdiction"]
    else:
        return lei["Entity.LegalJurisdiction"]
