:tocdepth: 5


.. image:: /_images/veros-logo-400px.png
   :align: center

|

Versatile Ocean Simulation in Pure Python
=========================================

Veros, *the versatile ocean simulator*, aims to be the swiss army knife of ocean modeling. It is a full-fledged :abbr:`GCM (general circulation model)` that supports anything between highly idealized configurations and realistic set-ups. Thanks to its interplay with `JAX <https://github.com/google/jax>`_, Veros runs efficiently on your laptop, gaming PC (including GPU support), or full-scale cluster. In short, we want to enable ocean modelling with a clear focus on simplicity, usability, and adaptability.

If you want to learn more about the background and capabilities of Veros, you should check out :doc:`quickstart/introduction`. If you are already convinced, you can jump right into action, and :doc:`learn how to get started <quickstart/get-started>` instead!

.. image:: /_images/tagline.png
   :scale: 50%
   :class: no-scaled-link
   :alt: ... because the Baroque is over.

.. toctree::
   :maxdepth: 2
   :caption: Quick Start

   quickstart/introduction
   quickstart/get-started

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorial/wave-propagation
   tutorial/dev
   tutorial/cluster

.. toctree::
   :maxdepth: 1
   :caption: Reference

   reference/setup-gallery
   reference/settings
   reference/variables
   reference/diagnostics
   reference/cli
   reference/public-api

.. toctree::
   :maxdepth: 1
   :caption: Plug-ins

   plugins/user-guide
   Biogeochemistry plug-in (Veros-BGC) <https://veros-bgc.readthedocs.io>
   plugins/developer-guide

.. toctree::
   :maxdepth: 2
   :caption: More Information

   more/faq
   more/benchmarks
   more/publications
   more/contact
   Visit us on GitHub <https://github.com/team-ocean/veros>
