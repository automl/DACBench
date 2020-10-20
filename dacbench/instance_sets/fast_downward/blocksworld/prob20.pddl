

(define (problem BW-rand-12)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 )
(:init
(arm-empty)
(on b1 b5)
(on b2 b7)
(on b3 b9)
(on b4 b11)
(on-table b5)
(on b6 b10)
(on b7 b4)
(on-table b8)
(on-table b9)
(on b10 b2)
(on b11 b8)
(on-table b12)
(clear b1)
(clear b3)
(clear b6)
(clear b12)
)
(:goal
(and
(on b1 b5)
(on b3 b1)
(on b5 b12)
(on b7 b3)
(on b9 b11)
(on b10 b9)
(on b11 b7)
(on b12 b8))
)
)


