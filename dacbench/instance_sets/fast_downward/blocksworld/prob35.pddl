

(define (problem BW-rand-13)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 )
(:init
(arm-empty)
(on b1 b6)
(on b2 b4)
(on b3 b2)
(on b4 b1)
(on b5 b11)
(on b6 b7)
(on b7 b10)
(on b8 b9)
(on b9 b13)
(on-table b10)
(on b11 b8)
(on-table b12)
(on b13 b12)
(clear b3)
(clear b5)
)
(:goal
(and
(on b1 b3)
(on b2 b5)
(on b3 b7)
(on b4 b1)
(on b5 b10)
(on b9 b8)
(on b10 b9)
(on b11 b12)
(on b13 b2))
)
)


