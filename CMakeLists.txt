# CMake entry point
cmake_minimum_required (VERSION 2.6)
project (MyClustering)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/Common)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# add "-O3" for library eigen optimization for matrix computation.
# "-O3" is C++ optimization flags (bigO 2)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3")

include_directories(
    /usr/include/
    /usr/include/eigen3
    /usr/local/include/
    /usr/local/include/vtk-6.3/
    src/Common/
	  ./src
    .
)

set(ALL_LIBS
	  common
)

add_definitions(
#	-DTW_STATIC
#	-DTW_NO_LIB_PRAGMA
#	-DTW_NO_DIRECT3D
#	-DGLEW_STATIC
#	-D_CRT_SECURE_NO_WARNINGS
)

###############################################
# generate dataGeneration executable
set(generation_SRC
	${CMAKE_CURRENT_SOURCE_DIR}/src/dataGeneration/generateData.cpp
)
add_executable(dataGeneration ${generation_SRC})
target_link_libraries(dataGeneration ${ALL_LIBS})

###############################################


###############################################
# generate sc executable
set(SC_SRC
    )
file( GLOB SC_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/spectralClustering/*.cpp )  
add_executable(sc ${SC_SRC})
target_link_libraries(sc ${ALL_LIBS})
###############################################


###############################################
# generate kmeans executable
set(KMEANS_SRC
    )
file( GLOB KMEANS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/kmeans/*.cpp )  
add_executable(kmeans ${KMEANS_SRC})
target_link_libraries(kmeans ${ALL_LIBS})
###############################################


###############################################
# generate kmedoids executable
set(KMEDOIDS_SRC
    )
file( GLOB KMEDOIDS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/kmedoids/*.cpp )  
add_executable(kmedoids ${KMEDOIDS_SRC})
target_link_libraries(kmedoids ${ALL_LIBS})
###############################################


###############################################
# generate ahc executable
set(AHC_SRC
    )
file( GLOB AHC_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/AHC/*.cpp )  
add_executable(ahc ${AHC_SRC})
target_link_libraries(ahc ${ALL_LIBS})
###############################################


###############################################
# generate dbscan executable
set(DBSCAN_SRC
    )
file( GLOB DBSCAN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/DBSCAN/*.cpp )  
add_executable(dbscan ${DBSCAN_SRC})
target_link_libraries(dbscan ${ALL_LIBS})
###############################################