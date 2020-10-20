

(define (problem BW-rand-12)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 )
(:init
(arm-empty)
(on b1 b12)
(on b2 b4)
(on b3 b2)
(on b4 b9)
(on b5 b1)
(on b6 b10)
(on b7 b8)
(on b8 b3)
(on-table b9)
(on b10 b5)
(on b11 b6)
(on b12 b7)
(clear b11)
)
(:goal
(and
(on b1 b10)
(on b3 b5)
(on b5 b2)
(on b6 b4)
(on b7 b11)
(on b8 b3)
(on b9 b12)
(on b10 b7)
(on b11 b6))
)
)


