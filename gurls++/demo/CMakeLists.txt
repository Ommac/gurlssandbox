# Copyright (C) 2011-2013  Istituto Italiano di Tecnologia, Massachussets Institute of Techology
# Authors: Elena Ceseracciu <elena.ceseracciu@iit.it>, Matteo Santoro <msantoro@mit.edu>
include_directories(${Gurls++_INCLUDE_DIRS})

#add_executable(GURLSloocvprimal GURLSloocvprimal.cpp)
#target_link_libraries(GURLSloocvprimal ${Gurls++_LIBRARIES})

#add_executable(helloworld helloworld.cpp)
#target_link_libraries(helloworld ${Gurls++_LIBRARIES})

#add_executable(traintest_demo traintest_demo.cpp)
#target_link_libraries(traintest_demo ${Gurls++_LIBRARIES})

#add_executable(sandbox sandbox.cpp)
#target_link_libraries(sandbox ${Gurls++_LIBRARIES})

#add_executable(recursiveRLS recursiveRLS.cpp)
#target_link_libraries(recursiveRLS ${Gurls++_LIBRARIES})

#add_executable(recursiveRLS recursiveRLS.cpp)
#target_link_libraries(recursiveRLS ${Gurls++_LIBRARIES})

#add_executable(recursiveRLS_id_chol recursiveRLS_id_chol.cpp)
#target_link_libraries(recursiveRLS_id_chol ${Gurls++_LIBRARIES})

# add_executable(subvecTest subvecTest.cpp)
# target_link_libraries(subvecTest ${Gurls++_LIBRARIES})

# add_executable(submatrixTest submatrixTest.cpp)
# target_link_libraries(submatrixTest ${Gurls++_LIBRARIES})

add_executable(subVecTest2 subVecTest2.cpp)
target_link_libraries(subVecTest2 ${Gurls++_LIBRARIES})

#add_executable(gp gp.cpp)
#target_link_libraries(gp ${Gurls++_LIBRARIES})
