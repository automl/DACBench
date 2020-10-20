

(define (problem BW-rand-13)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 )
(:init
(arm-empty)
(on-table b1)
(on-table b2)
(on b3 b2)
(on b4 b6)
(on b5 b1)
(on b6 b3)
(on b7 b4)
(on b8 b9)
(on b9 b11)
(on b10 b5)
(on b11 b7)
(on b12 b10)
(on b13 b12)
(clear b8)
(clear b13)
)
(:goal
(and
(on b1 b5)
(on b2 b3)
(on b5 b9)
(on b7 b6)
(on b9 b7)
(on b10 b8)
(on b11 b10)
(on b12 b4)
(on b13 b1))
)
)


