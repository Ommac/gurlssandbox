# Copyright (C) 2011-2013  Istituto Italiano di Tecnologia, Massachussets Institute of Techology
# Authors: Elena Ceseracciu <elena.ceseracciu@iit.it>, Matteo Santoro <msantoro@mit.edu>
include_directories(${Gurls++_INCLUDE_DIRS})

#add_executable(GURLSloocvprimal GURLSloocvprimal.cpp)
#target_link_libraries(GURLSloocvprimal ${Gurls++_LIBRARIES})

#add_executable(GURLSloocvprimal_kammo GURLSloocvprimal_kammo.cpp)
#target_link_libraries(GURLSloocvprimal_kammo ${Gurls++_LIBRARIES})

#add_executable(helloworld helloworld.cpp)
#target_link_libraries(helloworld ${Gurls++_LIBRARIES})

#add_executable(recursiveRLS recursiveRLS.cpp)
#target_link_libraries(recursiveRLS ${Gurls++_LIBRARIES})

#add_executable(recursiveRLS_id recursiveRLS_id.cpp)
#target_link_libraries(recursiveRLS_id ${Gurls++_LIBRARIES})

add_executable(recursiveRLS_id_chol recursiveRLS_id_chol.cpp)
target_link_libraries(recursiveRLS_id_chol ${Gurls++_LIBRARIES})

#add_executable(drot_test drot_test.cpp)
#target_link_libraries(drot_test ${Gurls++_LIBRARIES})

#add_executable(gp gp.cpp)
#target_link_libraries(gp ${Gurls++_LIBRARIES})
