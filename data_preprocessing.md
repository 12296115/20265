# Data Preprocessing and Dataset Validation

## Dataset Preparation

The phishing email dataset was systematically prepared before model development to improve data quality and ensure that the resulting experiments were based on valid and non-duplicated email records.

### Initial Dataset

- Initial records: 18,650
- Initial columns:
  - Email Text
  - Email Type
  - Unnamed: 0

### Removal of Unusable Records

Records that could not be used for modelling were removed.

- Records after unusable-record removal: 18,098
- Columns retained:
  - Email Text
  - Email Type

### Duplicate Removal

Duplicate email texts were identified using the `Email Text` field.

- Records before duplicate removal: 18,098
- Duplicate records identified: 577
- Records after duplicate removal: 17,521

The first occurrence of each duplicated email text was retained.

### Final Dataset Validation

The final cleaned dataset contains:

- Total records: 17,521
- Missing values: 0
- Duplicate email texts: 0
- Safe Email: 10,978
- Phishing Email: 6,543

The final class distribution was:

| Email Type | Records | Proportion |
|---|---:|---:|
| Safe Email | 10,978 | 62.66% |
| Phishing Email | 6,543 | 37.34% |
| **Total** | **17,521** | **100%** |

### Validation

The final dataset was validated for:

1. Missing values
2. Duplicate email texts
3. Class distribution
4. Dataset dimensions
5. Retained modelling columns

The validated dataset was subsequently used for train/test splitting and model development.

## Evidence

The preprocessing results were validated in Google Colab and documented through the project's progress evidence. The final dataset was not uploaded directly to GitHub because its file size exceeds the repository's standard web-upload limit.
