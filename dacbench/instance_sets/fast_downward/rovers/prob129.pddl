(define (problem roverprob) (:domain Rover)
(:objects
	general - Lander
	colour high_res low_res - Mode
	rover0 rover1 rover2 - Rover
	rover0store rover1store rover2store - Store
	waypoint0 waypoint1 waypoint2 waypoint3 waypoint4 waypoint5 waypoint6 waypoint7 waypoint8 waypoint9 waypoint10 waypoint11 waypoint12 waypoint13 waypoint14 waypoint15 waypoint16 waypoint17 waypoint18 waypoint19 waypoint20 waypoint21 waypoint22 waypoint23 waypoint24 - Waypoint
	camera0 camera1 camera2 - Camera
	objective0 objective1 objective2 objective3 objective4 objective5 objective6 objective7 - Objective
	)
(:init
	(visible waypoint0 waypoint6)
	(visible waypoint6 waypoint0)
	(visible waypoint0 waypoint10)
	(visible waypoint10 waypoint0)
	(visible waypoint0 waypoint18)
	(visible waypoint18 waypoint0)
	(visible waypoint0 waypoint20)
	(visible waypoint20 waypoint0)
	(visible waypoint1 waypoint0)
	(visible waypoint0 waypoint1)
	(visible waypoint1 waypoint2)
	(visible waypoint2 waypoint1)
	(visible waypoint1 waypoint4)
	(visible waypoint4 waypoint1)
	(visible waypoint1 waypoint6)
	(visible waypoint6 waypoint1)
	(visible waypoint1 waypoint9)
	(visible waypoint9 waypoint1)
	(visible waypoint1 waypoint14)
	(visible waypoint14 waypoint1)
	(visible waypoint2 waypoint9)
	(visible waypoint9 waypoint2)
	(visible waypoint3 waypoint5)
	(visible waypoint5 waypoint3)
	(visible waypoint3 waypoint7)
	(visible waypoint7 waypoint3)
	(visible waypoint3 waypoint8)
	(visible waypoint8 waypoint3)
	(visible waypoint3 waypoint15)
	(visible waypoint15 waypoint3)
	(visible waypoint4 waypoint0)
	(visible waypoint0 waypoint4)
	(visible waypoint4 waypoint6)
	(visible waypoint6 waypoint4)
	(visible waypoint4 waypoint7)
	(visible waypoint7 waypoint4)
	(visible waypoint4 waypoint19)
	(visible waypoint19 waypoint4)
	(visible waypoint4 waypoint21)
	(visible waypoint21 waypoint4)
	(visible waypoint5 waypoint4)
	(visible waypoint4 waypoint5)
	(visible waypoint6 waypoint2)
	(visible waypoint2 waypoint6)
	(visible waypoint6 waypoint8)
	(visible waypoint8 waypoint6)
	(visible waypoint6 waypoint9)
	(visible waypoint9 waypoint6)
	(visible waypoint7 waypoint6)
	(visible waypoint6 waypoint7)
	(visible waypoint7 waypoint13)
	(visible waypoint13 waypoint7)
	(visible waypoint8 waypoint2)
	(visible waypoint2 waypoint8)
	(visible waypoint8 waypoint9)
	(visible waypoint9 waypoint8)
	(visible waypoint8 waypoint11)
	(visible waypoint11 waypoint8)
	(visible waypoint8 waypoint17)
	(visible waypoint17 waypoint8)
	(visible waypoint8 waypoint19)
	(visible waypoint19 waypoint8)
	(visible waypoint9 waypoint0)
	(visible waypoint0 waypoint9)
	(visible waypoint9 waypoint11)
	(visible waypoint11 waypoint9)
	(visible waypoint9 waypoint12)
	(visible waypoint12 waypoint9)
	(visible waypoint9 waypoint17)
	(visible waypoint17 waypoint9)
	(visible waypoint9 waypoint21)
	(visible waypoint21 waypoint9)
	(visible waypoint10 waypoint4)
	(visible waypoint4 waypoint10)
	(visible waypoint10 waypoint5)
	(visible waypoint5 waypoint10)
	(visible waypoint10 waypoint19)
	(visible waypoint19 waypoint10)
	(visible waypoint10 waypoint22)
	(visible waypoint22 waypoint10)
	(visible waypoint10 waypoint24)
	(visible waypoint24 waypoint10)
	(visible waypoint11 waypoint1)
	(visible waypoint1 waypoint11)
	(visible waypoint11 waypoint7)
	(visible waypoint7 waypoint11)
	(visible waypoint11 waypoint16)
	(visible waypoint16 waypoint11)
	(visible waypoint12 waypoint6)
	(visible waypoint6 waypoint12)
	(visible waypoint12 waypoint14)
	(visible waypoint14 waypoint12)
	(visible waypoint14 waypoint7)
	(visible waypoint7 waypoint14)
	(visible waypoint14 waypoint11)
	(visible waypoint11 waypoint14)
	(visible waypoint14 waypoint24)
	(visible waypoint24 waypoint14)
	(visible waypoint15 waypoint0)
	(visible waypoint0 waypoint15)
	(visible waypoint15 waypoint5)
	(visible waypoint5 waypoint15)
	(visible waypoint15 waypoint11)
	(visible waypoint11 waypoint15)
	(visible waypoint15 waypoint16)
	(visible waypoint16 waypoint15)
	(visible waypoint15 waypoint19)
	(visible waypoint19 waypoint15)
	(visible waypoint16 waypoint2)
	(visible waypoint2 waypoint16)
	(visible waypoint16 waypoint17)
	(visible waypoint17 waypoint16)
	(visible waypoint16 waypoint22)
	(visible waypoint22 waypoint16)
	(visible waypoint17 waypoint2)
	(visible waypoint2 waypoint17)
	(visible waypoint17 waypoint6)
	(visible waypoint6 waypoint17)
	(visible waypoint17 waypoint11)
	(visible waypoint11 waypoint17)
	(visible waypoint17 waypoint15)
	(visible waypoint15 waypoint17)
	(visible waypoint18 waypoint3)
	(visible waypoint3 waypoint18)
	(visible waypoint18 waypoint15)
	(visible waypoint15 waypoint18)
	(visible waypoint19 waypoint5)
	(visible waypoint5 waypoint19)
	(visible waypoint19 waypoint14)
	(visible waypoint14 waypoint19)
	(visible waypoint19 waypoint17)
	(visible waypoint17 waypoint19)
	(visible waypoint20 waypoint17)
	(visible waypoint17 waypoint20)
	(visible waypoint20 waypoint22)
	(visible waypoint22 waypoint20)
	(visible waypoint21 waypoint1)
	(visible waypoint1 waypoint21)
	(visible waypoint21 waypoint2)
	(visible waypoint2 waypoint21)
	(visible waypoint21 waypoint6)
	(visible waypoint6 waypoint21)
	(visible waypoint21 waypoint8)
	(visible waypoint8 waypoint21)
	(visible waypoint21 waypoint10)
	(visible waypoint10 waypoint21)
	(visible waypoint21 waypoint15)
	(visible waypoint15 waypoint21)
	(visible waypoint21 waypoint18)
	(visible waypoint18 waypoint21)
	(visible waypoint21 waypoint22)
	(visible waypoint22 waypoint21)
	(visible waypoint22 waypoint0)
	(visible waypoint0 waypoint22)
	(visible waypoint22 waypoint8)
	(visible waypoint8 waypoint22)
	(visible waypoint22 waypoint18)
	(visible waypoint18 waypoint22)
	(visible waypoint23 waypoint20)
	(visible waypoint20 waypoint23)
	(visible waypoint24 waypoint2)
	(visible waypoint2 waypoint24)
	(visible waypoint24 waypoint8)
	(visible waypoint8 waypoint24)
	(visible waypoint24 waypoint15)
	(visible waypoint15 waypoint24)
	(visible waypoint24 waypoint16)
	(visible waypoint16 waypoint24)
	(visible waypoint24 waypoint23)
	(visible waypoint23 waypoint24)
	(at_rock_sample waypoint0)
	(at_soil_sample waypoint1)
	(at_rock_sample waypoint1)
	(at_soil_sample waypoint3)
	(at_rock_sample waypoint3)
	(at_rock_sample waypoint4)
	(at_soil_sample waypoint5)
	(at_rock_sample waypoint6)
	(at_rock_sample waypoint7)
	(at_soil_sample waypoint9)
	(at_soil_sample waypoint10)
	(at_soil_sample waypoint11)
	(at_soil_sample waypoint13)
	(at_soil_sample waypoint14)
	(at_soil_sample waypoint15)
	(at_rock_sample waypoint15)
	(at_soil_sample waypoint16)
	(at_rock_sample waypoint16)
	(at_soil_sample waypoint17)
	(at_soil_sample waypoint18)
	(at_rock_sample waypoint18)
	(at_soil_sample waypoint19)
	(at_rock_sample waypoint19)
	(at_rock_sample waypoint20)
	(at_soil_sample waypoint21)
	(at_rock_sample waypoint21)
	(at_soil_sample waypoint22)
	(at_lander general waypoint17)
	(channel_free general)
	(at rover0 waypoint21)
	(available rover0)
	(store_of rover0store rover0)
	(empty rover0store)
	(equipped_for_soil_analysis rover0)
	(equipped_for_imaging rover0)
	(can_traverse rover0 waypoint21 waypoint1)
	(can_traverse rover0 waypoint1 waypoint21)
	(can_traverse rover0 waypoint21 waypoint2)
	(can_traverse rover0 waypoint2 waypoint21)
	(can_traverse rover0 waypoint21 waypoint4)
	(can_traverse rover0 waypoint4 waypoint21)
	(can_traverse rover0 waypoint21 waypoint8)
	(can_traverse rover0 waypoint8 waypoint21)
	(can_traverse rover0 waypoint21 waypoint10)
	(can_traverse rover0 waypoint10 waypoint21)
	(can_traverse rover0 waypoint21 waypoint15)
	(can_traverse rover0 waypoint15 waypoint21)
	(can_traverse rover0 waypoint21 waypoint18)
	(can_traverse rover0 waypoint18 waypoint21)
	(can_traverse rover0 waypoint1 waypoint0)
	(can_traverse rover0 waypoint0 waypoint1)
	(can_traverse rover0 waypoint1 waypoint6)
	(can_traverse rover0 waypoint6 waypoint1)
	(can_traverse rover0 waypoint1 waypoint11)
	(can_traverse rover0 waypoint11 waypoint1)
	(can_traverse rover0 waypoint2 waypoint9)
	(can_traverse rover0 waypoint9 waypoint2)
	(can_traverse rover0 waypoint2 waypoint16)
	(can_traverse rover0 waypoint16 waypoint2)
	(can_traverse rover0 waypoint2 waypoint17)
	(can_traverse rover0 waypoint17 waypoint2)
	(can_traverse rover0 waypoint4 waypoint5)
	(can_traverse rover0 waypoint5 waypoint4)
	(can_traverse rover0 waypoint8 waypoint3)
	(can_traverse rover0 waypoint3 waypoint8)
	(can_traverse rover0 waypoint8 waypoint19)
	(can_traverse rover0 waypoint19 waypoint8)
	(can_traverse rover0 waypoint8 waypoint22)
	(can_traverse rover0 waypoint22 waypoint8)
	(can_traverse rover0 waypoint15 waypoint24)
	(can_traverse rover0 waypoint24 waypoint15)
	(can_traverse rover0 waypoint0 waypoint20)
	(can_traverse rover0 waypoint20 waypoint0)
	(can_traverse rover0 waypoint6 waypoint7)
	(can_traverse rover0 waypoint7 waypoint6)
	(can_traverse rover0 waypoint6 waypoint12)
	(can_traverse rover0 waypoint12 waypoint6)
	(can_traverse rover0 waypoint24 waypoint14)
	(can_traverse rover0 waypoint14 waypoint24)
	(can_traverse rover0 waypoint24 waypoint23)
	(can_traverse rover0 waypoint23 waypoint24)
	(can_traverse rover0 waypoint7 waypoint13)
	(can_traverse rover0 waypoint13 waypoint7)
	(at rover1 waypoint7)
	(available rover1)
	(store_of rover1store rover1)
	(empty rover1store)
	(equipped_for_imaging rover1)
	(can_traverse rover1 waypoint7 waypoint3)
	(can_traverse rover1 waypoint3 waypoint7)
	(can_traverse rover1 waypoint7 waypoint6)
	(can_traverse rover1 waypoint6 waypoint7)
	(can_traverse rover1 waypoint7 waypoint11)
	(can_traverse rover1 waypoint11 waypoint7)
	(can_traverse rover1 waypoint7 waypoint13)
	(can_traverse rover1 waypoint13 waypoint7)
	(can_traverse rover1 waypoint7 waypoint14)
	(can_traverse rover1 waypoint14 waypoint7)
	(can_traverse rover1 waypoint3 waypoint5)
	(can_traverse rover1 waypoint5 waypoint3)
	(can_traverse rover1 waypoint3 waypoint8)
	(can_traverse rover1 waypoint8 waypoint3)
	(can_traverse rover1 waypoint3 waypoint18)
	(can_traverse rover1 waypoint18 waypoint3)
	(can_traverse rover1 waypoint6 waypoint1)
	(can_traverse rover1 waypoint1 waypoint6)
	(can_traverse rover1 waypoint6 waypoint2)
	(can_traverse rover1 waypoint2 waypoint6)
	(can_traverse rover1 waypoint6 waypoint9)
	(can_traverse rover1 waypoint9 waypoint6)
	(can_traverse rover1 waypoint6 waypoint12)
	(can_traverse rover1 waypoint12 waypoint6)
	(can_traverse rover1 waypoint11 waypoint15)
	(can_traverse rover1 waypoint15 waypoint11)
	(can_traverse rover1 waypoint11 waypoint16)
	(can_traverse rover1 waypoint16 waypoint11)
	(can_traverse rover1 waypoint11 waypoint17)
	(can_traverse rover1 waypoint17 waypoint11)
	(can_traverse rover1 waypoint14 waypoint19)
	(can_traverse rover1 waypoint19 waypoint14)
	(can_traverse rover1 waypoint14 waypoint24)
	(can_traverse rover1 waypoint24 waypoint14)
	(can_traverse rover1 waypoint8 waypoint22)
	(can_traverse rover1 waypoint22 waypoint8)
	(can_traverse rover1 waypoint18 waypoint0)
	(can_traverse rover1 waypoint0 waypoint18)
	(can_traverse rover1 waypoint18 waypoint21)
	(can_traverse rover1 waypoint21 waypoint18)
	(can_traverse rover1 waypoint1 waypoint4)
	(can_traverse rover1 waypoint4 waypoint1)
	(can_traverse rover1 waypoint19 waypoint10)
	(can_traverse rover1 waypoint10 waypoint19)
	(can_traverse rover1 waypoint24 waypoint23)
	(can_traverse rover1 waypoint23 waypoint24)
	(can_traverse rover1 waypoint22 waypoint20)
	(can_traverse rover1 waypoint20 waypoint22)
	(at rover2 waypoint12)
	(available rover2)
	(store_of rover2store rover2)
	(empty rover2store)
	(equipped_for_rock_analysis rover2)
	(equipped_for_imaging rover2)
	(can_traverse rover2 waypoint12 waypoint6)
	(can_traverse rover2 waypoint6 waypoint12)
	(can_traverse rover2 waypoint12 waypoint9)
	(can_traverse rover2 waypoint9 waypoint12)
	(can_traverse rover2 waypoint12 waypoint14)
	(can_traverse rover2 waypoint14 waypoint12)
	(can_traverse rover2 waypoint6 waypoint0)
	(can_traverse rover2 waypoint0 waypoint6)
	(can_traverse rover2 waypoint6 waypoint1)
	(can_traverse rover2 waypoint1 waypoint6)
	(can_traverse rover2 waypoint6 waypoint2)
	(can_traverse rover2 waypoint2 waypoint6)
	(can_traverse rover2 waypoint6 waypoint4)
	(can_traverse rover2 waypoint4 waypoint6)
	(can_traverse rover2 waypoint6 waypoint7)
	(can_traverse rover2 waypoint7 waypoint6)
	(can_traverse rover2 waypoint6 waypoint17)
	(can_traverse rover2 waypoint17 waypoint6)
	(can_traverse rover2 waypoint6 waypoint21)
	(can_traverse rover2 waypoint21 waypoint6)
	(can_traverse rover2 waypoint14 waypoint11)
	(can_traverse rover2 waypoint11 waypoint14)
	(can_traverse rover2 waypoint14 waypoint19)
	(can_traverse rover2 waypoint19 waypoint14)
	(can_traverse rover2 waypoint14 waypoint24)
	(can_traverse rover2 waypoint24 waypoint14)
	(can_traverse rover2 waypoint0 waypoint10)
	(can_traverse rover2 waypoint10 waypoint0)
	(can_traverse rover2 waypoint0 waypoint15)
	(can_traverse rover2 waypoint15 waypoint0)
	(can_traverse rover2 waypoint0 waypoint20)
	(can_traverse rover2 waypoint20 waypoint0)
	(can_traverse rover2 waypoint0 waypoint22)
	(can_traverse rover2 waypoint22 waypoint0)
	(can_traverse rover2 waypoint2 waypoint8)
	(can_traverse rover2 waypoint8 waypoint2)
	(can_traverse rover2 waypoint4 waypoint5)
	(can_traverse rover2 waypoint5 waypoint4)
	(can_traverse rover2 waypoint7 waypoint13)
	(can_traverse rover2 waypoint13 waypoint7)
	(can_traverse rover2 waypoint21 waypoint18)
	(can_traverse rover2 waypoint18 waypoint21)
	(on_board camera0 rover0)
	(calibration_target camera0 objective4)
	(supports camera0 colour)
	(on_board camera1 rover1)
	(calibration_target camera1 objective0)
	(supports camera1 high_res)
	(on_board camera2 rover2)
	(calibration_target camera2 objective5)
	(supports camera2 high_res)
	(supports camera2 low_res)
	(visible_from objective0 waypoint0)
	(visible_from objective0 waypoint1)
	(visible_from objective0 waypoint2)
	(visible_from objective0 waypoint3)
	(visible_from objective0 waypoint4)
	(visible_from objective0 waypoint5)
	(visible_from objective0 waypoint6)
	(visible_from objective0 waypoint7)
	(visible_from objective0 waypoint8)
	(visible_from objective0 waypoint9)
	(visible_from objective0 waypoint10)
	(visible_from objective0 waypoint11)
	(visible_from objective0 waypoint12)
	(visible_from objective0 waypoint13)
	(visible_from objective0 waypoint14)
	(visible_from objective0 waypoint15)
	(visible_from objective0 waypoint16)
	(visible_from objective1 waypoint0)
	(visible_from objective1 waypoint1)
	(visible_from objective1 waypoint2)
	(visible_from objective1 waypoint3)
	(visible_from objective1 waypoint4)
	(visible_from objective1 waypoint5)
	(visible_from objective1 waypoint6)
	(visible_from objective1 waypoint7)
	(visible_from objective1 waypoint8)
	(visible_from objective1 waypoint9)
	(visible_from objective1 waypoint10)
	(visible_from objective1 waypoint11)
	(visible_from objective1 waypoint12)
	(visible_from objective1 waypoint13)
	(visible_from objective1 waypoint14)
	(visible_from objective1 waypoint15)
	(visible_from objective2 waypoint0)
	(visible_from objective2 waypoint1)
	(visible_from objective2 waypoint2)
	(visible_from objective2 waypoint3)
	(visible_from objective2 waypoint4)
	(visible_from objective3 waypoint0)
	(visible_from objective3 waypoint1)
	(visible_from objective3 waypoint2)
	(visible_from objective3 waypoint3)
	(visible_from objective3 waypoint4)
	(visible_from objective3 waypoint5)
	(visible_from objective3 waypoint6)
	(visible_from objective3 waypoint7)
	(visible_from objective3 waypoint8)
	(visible_from objective3 waypoint9)
	(visible_from objective3 waypoint10)
	(visible_from objective3 waypoint11)
	(visible_from objective3 waypoint12)
	(visible_from objective3 waypoint13)
	(visible_from objective3 waypoint14)
	(visible_from objective3 waypoint15)
	(visible_from objective3 waypoint16)
	(visible_from objective3 waypoint17)
	(visible_from objective3 waypoint18)
	(visible_from objective3 waypoint19)
	(visible_from objective3 waypoint20)
	(visible_from objective3 waypoint21)
	(visible_from objective3 waypoint22)
	(visible_from objective3 waypoint23)
	(visible_from objective4 waypoint0)
	(visible_from objective4 waypoint1)
	(visible_from objective4 waypoint2)
	(visible_from objective4 waypoint3)
	(visible_from objective4 waypoint4)
	(visible_from objective4 waypoint5)
	(visible_from objective4 waypoint6)
	(visible_from objective4 waypoint7)
	(visible_from objective4 waypoint8)
	(visible_from objective4 waypoint9)
	(visible_from objective4 waypoint10)
	(visible_from objective4 waypoint11)
	(visible_from objective4 waypoint12)
	(visible_from objective4 waypoint13)
	(visible_from objective4 waypoint14)
	(visible_from objective4 waypoint15)
	(visible_from objective4 waypoint16)
	(visible_from objective5 waypoint0)
	(visible_from objective6 waypoint0)
	(visible_from objective6 waypoint1)
	(visible_from objective6 waypoint2)
	(visible_from objective6 waypoint3)
	(visible_from objective6 waypoint4)
	(visible_from objective6 waypoint5)
	(visible_from objective7 waypoint0)
)

(:goal (and
(communicated_soil_data waypoint18)
(communicated_soil_data waypoint5)
(communicated_soil_data waypoint11)
(communicated_soil_data waypoint14)
(communicated_rock_data waypoint1)
(communicated_rock_data waypoint15)
(communicated_rock_data waypoint20)
(communicated_rock_data waypoint0)
(communicated_image_data objective6 high_res)
(communicated_image_data objective3 high_res)
(communicated_image_data objective2 high_res)
(communicated_image_data objective1 high_res)
	)
)
)
