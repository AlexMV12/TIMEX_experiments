This package contains the code to run the tests for "TIMEX: A Framework for Time-Series Forecasting-as-a-Service".
All the code has been tested on a Linux machine, with Python 3.9; however, Python 3.8 should work too.


This folder contains:
  - source code for TIMEX (in folder timexseries);
  - three folders containing the scripts and datasets for the three scenarios considered:
    - Studio-Covid19: Covid-19 scenario in Italy.
    - Studio-GOOG: Google stock closing price scenario.
    - Studio-Monthly-Milk-Production: Montly milk production per cow scenario.
  - two scripts to install Darts and TIMEX in separated freshly created Python virtual environments;
  - three scripts to launch both TIMEX tests and Darts tests, one for each scenario;
  - three scripts to launch both TIMEX tests and Darts tests, one for each scenario, using Docker;
  - a script to build a Docker image;
  - a Poetry pyproject file (if you don't know what this means, feel free to ignore this);
  - this readme file;
  - some screenshots with proofs;
  - results table.
  

To run the tests, two options are available:

1. Native
This way will install two Python virtual-environments with TIMEX and Darts.
Then, the tests can be launched using the provided scripts.

Steps:
  1. Run `install_timex_venv.sh` and then run `install_darts_venv.sh`.
  2. Run `launch_milk_tests.sh`, `launch_covid_tests.sh`, `launch_goog_tests.sh`.

The result will be printed on the console. Note that tests can require hours; in particular Covid-19 is the most expensive.
     
2. Docker
This will create a Docker image, based on Debian, on which all the dependencies will be installed.
It can be useful if there are problems installing Facebook Prophet/Torch on the bare metal machine.

Steps:
  1. Run `create_docker_image.sh`.
  2. Run `launch_milk_tests_docker.sh`, `launch_covid_tests_docker.sh`, `launch_goog_tests_docker.sh`.

The result will be printed on the console. Note that tests can require hours; in particular Covid-19 is the most expensive.

