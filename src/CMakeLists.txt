#Copyright(c) Microsoft Corporation.All rights reserved.
#Licensed under the MIT                        license.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

# Enable CUDA support
find_package(CUDA REQUIRED)

# Include CUDA headers and libraries
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

# Enable GPU support for the project
option(ENABLE_GPU "Enable GPU acceleration" ON)

if (ENABLE_GPU)
    add_definitions(-DUSE_GPU)
    message(STATUS "GPU acceleration is enabled.")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -arch=sm_70") # Adjust arch based on your GPU
    list(APPEND EXTRA_LIBS ${CUDA_LIBRARIES} ${CUDA_CUDART_LIBRARY})
else()
    message(STATUS "GPU acceleration is disabled.")
endif()

if(MSVC)
    add_subdirectory(dll)
else()
    #file(GLOB CPP_SOURCES *.cpp)
    set(CPP_SOURCES abstract_data_store.cpp ann_exception.cpp disk_utils.cpp 
        distance.cpp index.cpp in_mem_graph_store.cpp in_mem_data_store.cpp
        linux_aligned_file_reader.cpp math_utils.cpp natural_number_map.cpp
        in_mem_data_store.cpp in_mem_graph_store.cpp
        natural_number_set.cpp memory_mapper.cpp partition.cpp pq.cpp
        pq_flash_index.cpp scratch.cpp logger.cpp utils.cpp filter_utils.cpp index_factory.cpp abstract_index.cpp pq_l2_distance.cpp pq_data_store.cpp gpu_kernels.cu)
    if (RESTAPI)
        list(APPEND CPP_SOURCES restapi/search_wrapper.cpp restapi/server.cpp)
    endif()

    set(CUDA_SOURCES gpu_kernels.cu)
    set_source_files_properties(${PROJECT_NAME} ${PROJECT_NAME}_s PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)

    CUDA_ADD_LIBRARY(${PROJECT_NAME} ${CPP_SOURCES})
    CUDA_ADD_LIBRARY(${PROJECT_NAME}_s STATIC ${CPP_SOURCES})

    # target_link_libraries(${PROJECT_NAME} PUBLIC ${CUDA_LIBRARIES})
    # target_link_libraries(${PROJECT_NAME}_s PUBLIC ${CUDA_LIBRARIES})

    # set_target_properties(${PROJECT_NAME} ${PROJECT_NAME}_s PROPERTIES
    #     CUDA_SEPARABLE_COMPILATION ON
    #     CUDA_STANDARD 17
    # )

endif()

if (NOT MSVC)
    install(TARGETS ${PROJECT_NAME} LIBRARY)
endif()
