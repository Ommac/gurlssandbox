FIND_PATH(ATLAS_INCLUDE_DIR cblas.h  /usr/include /usr/local/include /usr/local/include/atlas)

SET(ATLAS_LIB_SEARCH_DIRS /usr/lib/sse2 /usr/lib/atlas/sse2 /usr/local/lib/atlas /usr/lib/atlas-base)
	
FIND_LIBRARY( ATLAS_LIBS NAMES atlas PATHS ${ATLAS_LIB_SEARCH_DIRS})
FIND_LIBRARY( ATLAS_BLAS_LIBS NAMES f77blas ptf77blas PATHS ${ATLAS_LIB_SEARCH_DIRS})
#FIND_LIBRARY( ATLAS_CBLAS_LIBS NAMES cblas ptcblas PATHS ${ATLAS_LIB_SEARCH_DIRS})
FIND_LIBRARY( ATLAS_LAPACK_LIBS NAMES lapack PATHS ${ATLAS_LIB_SEARCH_DIRS})

SET( ATLAS_INCLUDE_FILE ${ATLAS_INCLUDE_DIR}/cblas.h )

IF (ATLAS_INCLUDE_DIR AND ATLAS_BLAS_LIBS)
  SET(ATLAS_FOUND ON)
ENDIF (ATLAS_INCLUDE_DIR AND ATLAS_BLAS_LIBS)
	
IF (ATLAS_FOUND)
  IF (NOT ATLAS_FIND_QUIETLY)
     MESSAGE(STATUS "Found ATLAS: ${ATLAS_INCLUDE_DIR}")
  ENDIF (NOT ATLAS_FIND_QUIETLY)
ELSE(ATLAS_FOUND)
  IF (ATLAS_FIND_REQUIRED)
     MESSAGE(FATAL_ERROR "Could not find ATLAS")
  ENDIF (ATLAS_FIND_REQUIRED)
ENDIF (ATLAS_FOUND)

set(ATLAS_INCLUDE_DIRS ${ATLAS_INCLUDE_DIR})
