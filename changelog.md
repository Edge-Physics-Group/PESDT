10 Nov 2026 PESDT 2.0.5.2
- Fix bug in Calc stark ne due to change in spec_line_dict format

xx Aug 2026 PESDT 2.0.5.2
- Fix bug in process.py when calling "define_Hydrogenic_plasma_model" for molecules


2 Aug 2026 PESDT 2.0.5.1
- First implementation of Multi-species plasmas, and impurities
  - Multi-species bolometry: No FF or FF-FB, because the approximations used in
  continuolib are not accurate enough for Z>2
    - Separated into sources per charge state in the order 0 to MAX_CHARGE
  - Multi-species emission: Only ADAS as data source
- He plasmas not supported yet, need to modify plasma reading, such that it does
  not read e.g. hydrogenic molecular quantities when they are not present

29 July 2026 PESDT 2.0.5.0
- Re-activated OpenADAS as a data source
- Added bolometry
  - ADAS uses plt and prb, ff and ff-fb from continuolib
  - AMJUEL sums up all available hydrogenic lines
  - Has not been validated, use with caution
  - For JET need to check that origin is not blocked
- Main view is now separated into line-emission diagnostics, cameras, bolometry, and other diagnostics
  - Currently other diagnostics is a placeholder, none have been implemented yet
