

(define (problem BW-rand-14)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 )
(:init
(arm-empty)
(on b1 b3)
(on b2 b1)
(on b3 b4)
(on b4 b12)
(on b5 b7)
(on b6 b8)
(on b7 b13)
(on-table b8)
(on b9 b14)
(on b10 b11)
(on b11 b2)
(on b12 b9)
(on b13 b10)
(on-table b14)
(clear b5)
(clear b6)
)
(:goal
(and
(on b1 b5)
(on b2 b9)
(on b5 b10)
(on b6 b13)
(on b7 b4)
(on b8 b1)
(on b9 b8)
(on b13 b11)
(on b14 b3))
)
)


