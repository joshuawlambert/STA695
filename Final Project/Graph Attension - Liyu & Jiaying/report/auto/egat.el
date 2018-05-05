(TeX-add-style-hook
 "egat"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("IEEEtran" "conference")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("babel" "english")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "IEEEtran"
    "IEEEtran10"
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
    "amssymb"
    "multicol"
    "lipsum")
   (LaTeX-add-labels
    "sec:introduction"
    "sec:related-works"
    "fig:gat"
    "sec:architecture"
    "sec:graph-attention"
    "eq:gat"
    "eq:gat-average"
    "sec:graph-attention-with"
    "fig:egat"
    "sec:edge-latents-guided"
    "eq:1"
    "eq:egat"
    "sec:experimental-results"
    "sec:dataset"
    "tab:cora"
    "sec:performance"
    "tab:result"
    "sec:concl-future-direct")
   (LaTeX-add-bibliographies
    "ref"))
 :latex)

