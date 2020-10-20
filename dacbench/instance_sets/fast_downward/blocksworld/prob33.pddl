

(define (problem BW-rand-13)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 )
(:init
(arm-empty)
(on b1 b7)
(on-table b2)
(on b3 b10)
(on b4 b11)
(on b5 b13)
(on b6 b12)
(on b7 b6)
(on b8 b4)
(on b9 b3)
(on b10 b2)
(on b11 b1)
(on-table b12)
(on b13 b8)
(clear b5)
(clear b9)
)
(:goal
(and
(on b1 b12)
(on b2 b7)
(on b3 b13)
(on b6 b1)
(on b7 b6)
(on b9 b3)
(on b10 b11)
(on b11 b2)
(on b12 b5)
(on b13 b4))
)
)


