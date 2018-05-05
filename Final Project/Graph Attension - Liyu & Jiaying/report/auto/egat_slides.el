(TeX-add-style-hook
 "egat_slides"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("beamer" "compress")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("babel" "english")))
   (add-to-list 'LaTeX-verbatim-environments-local "semiverbatim")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "beamer"
    "beamer10"
    "graphicx"
    "graphics"
    "hyperref"
    "color"
    "babel"
    "subfigure"
    "verbatim"
    "isomath"
    "commath"
    "amsmath"
    "amssymb")
   (TeX-add-symbols
    "vec")
   (LaTeX-add-labels
    "tab:cora"
    "tab:result")
   (LaTeX-add-environments
    "wideitemize")
   (LaTeX-add-bibliographies
    "ref"))
 :latex)

