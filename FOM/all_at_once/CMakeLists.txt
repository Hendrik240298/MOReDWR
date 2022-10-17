##
#  CMake script for the space time finite elements program:
##

# Set the name of the project and target:
SET(TARGET "main")

SET(TARGET_SRC
  ${TARGET}.cc
  )

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

# Usually, you will not need to modify anything beyond this point...
IF(EXISTS /home/ifam/fischer/dealii-candi/deal.II-v9.2.0)
    CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)
    set (DEAL_II_DIR  /home/ifam/fischer/dealii-candi/deal.II-v9.2.0)
    set (DEAL_II_PATH  /home/ifam/fischer/dealii-candi/deal.II-v9.2.0)

    FIND_PACKAGE(deal.II 9.2.0 EXACT
        HINTS ${DEAL_II_DIR}
    )
ELSEIF(EXISTS /home/hendrik/dealii-candi/deal.II-v9.3.1)
     CMAKE_MINIMUM_REQUIRED(VERSION 3.1.0)
     set (DEAL_II_DIR  /home/hendrik/dealii-candi/deal.II-v9.3.1)
     set (DEAL_II_PATH  /home/hendrik/dealii-candi/deal.II-v9.3.1)

     FIND_PACKAGE(deal.II 9.3.0
         HINTS ${DEAL_II_DIR} /home/hendrik/dealii-candi/
     )
ENDIF()

IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()


FIND_PACKAGE(deal.II 9.0 QUIET
  HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})
DEAL_II_INVOKE_AUTOPILOT()

