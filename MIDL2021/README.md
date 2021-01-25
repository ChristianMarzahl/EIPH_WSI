# WSI Registrierung

## Anspruch der Registrierung

- WSI
  - Cyto
  - Tissue
- Unterschiedliche Scanner


## Idee

- Low level features like 
  - ORB 
  - SIFT 
  - SURF
  - Enhanced Correlation Coefficient (ECC), 
  - Fast Fourier Transform (FFT)
- Hight level features like object detection
- Quadtree Source Target 
  - Registrierung auf jedem Level verbessern
  - Nur bereiche die interessieren höher auflösen
  - Closest point registration with outliner filtering

## Vorteile Tree

- Tiefe individuell festlegbar
- Schnell und kein DL
- Fehler von höheren Leveln werden immer mehr reduziert
- Fokusierbereiche können vorgegeben werden anhand von GT Punkten
- 3D erweiterbar


# Alternative Paper:

## [PathFlow-MixMatch for Whole Slide Image Registration: An Investigation of a Segment-Based Scalable Image Registration Method](https://www.biorxiv.org/content/10.1101/2020.03.22.002402v1)

### Idee

- WSI in bereiche mit Gewebe unterteilen.
- Und diese bereiche von Interesse registrieren

### Limitations

- Nur eine Gewebeart benutzt (Leber)
- Im durchschnitt 80 Pixel daneben

## [Robust hierarchical density estimation and regression for re-stained histological whole slide image co-registration](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0220074)

### Idee

- refinements to existing image registration methods (patches)
- effective weighting strategy using kernel density estimation to
mitigate registration errors
  - Für putliner removal
- e linear relationship across WSI levels to
improve accuracy


### Limitentions

- Keine afine transformation (nur rigid)
- Gleicher Scanner
- Hierarchical resolution regression
  - Zeigt vielleicht nicht den gleichen Patch daher hoher level 0 fehler


### Code

- “imreg_dft”
- https://github.com/smujiang/Re-stained_WSIs_Registration


## [Dynamic registration for gigapixel serial whole slide images](https://ieeexplore.ieee.org/abstract/document/7950552?casa_token=TeA2BoD3iQcAAAAA:GmAX1g9V3mJwkPXPGSDVwfsrP_uwPB63B36fNO6Jn6y5Xgtq1yqvqfOVoish-ZlZ0asp9hxE0g)

### Idee

- low resolution pre-alignment, global and local registration 
- multi-resolution transformation mapping
- high resolution registration propagation 