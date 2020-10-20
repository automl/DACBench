

(define (problem BW-rand-12)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 )
(:init
(arm-empty)
(on b1 b9)
(on b2 b11)
(on b3 b4)
(on b4 b7)
(on b5 b10)
(on b6 b8)
(on b7 b1)
(on b8 b5)
(on b9 b2)
(on b10 b12)
(on b11 b6)
(on-table b12)
(clear b3)
)
(:goal
(and
(on b1 b11)
(on b2 b4)
(on b3 b7)
(on b4 b10)
(on b6 b5)
(on b7 b1)
(on b9 b3)
(on b11 b6))
)
)


