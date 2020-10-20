

(define (problem BW-rand-12)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 )
(:init
(arm-empty)
(on b1 b5)
(on b2 b4)
(on b3 b11)
(on b4 b1)
(on b5 b12)
(on b6 b3)
(on-table b7)
(on-table b8)
(on-table b9)
(on b10 b7)
(on b11 b8)
(on b12 b9)
(clear b2)
(clear b6)
(clear b10)
)
(:goal
(and
(on b2 b6)
(on b3 b2)
(on b5 b12)
(on b6 b1)
(on b8 b5)
(on b9 b4)
(on b10 b3)
(on b11 b8))
)
)


