

(define (problem BW-rand-19)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 )
(:init
(arm-empty)
(on b1 b15)
(on-table b2)
(on b3 b7)
(on b4 b2)
(on b5 b12)
(on b6 b8)
(on b7 b1)
(on-table b8)
(on b9 b3)
(on b10 b6)
(on b11 b17)
(on b12 b9)
(on b13 b10)
(on-table b14)
(on b15 b4)
(on-table b16)
(on-table b17)
(on b18 b16)
(on-table b19)
(clear b5)
(clear b11)
(clear b13)
(clear b14)
(clear b18)
(clear b19)
)
(:goal
(and
(on b1 b9)
(on b2 b13)
(on b3 b2)
(on b5 b3)
(on b6 b4)
(on b8 b19)
(on b9 b16)
(on b10 b18)
(on b11 b6)
(on b12 b14)
(on b14 b8)
(on b15 b5)
(on b16 b12)
(on b17 b10)
(on b19 b17))
)
)


