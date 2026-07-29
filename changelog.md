29 July 2026 PESDT 2.05a
- Re-activated OpenADAS as a data source
- Added bolometry
  - ADAS uses plt and prb, ff and ff-fb from continuolib
  - AMJUEL sums up all available hydrogenic lines
  - Has not been validated, use with caution
  - For JET need to check that origin is not blocked
- Main view is now separated into line-emission diagnostics, cameras, bolometry, and other diagnostics
  - Currently other diagnostics is a placeholder, none have been implemented yet