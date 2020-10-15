

(define (problem BW-rand-12)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 )
(:init
(arm-empty)
(on-table b1)
(on b2 b5)
(on-table b3)
(on-table b4)
(on b5 b12)
(on b6 b11)
(on b7 b4)
(on b8 b2)
(on b9 b6)
(on-table b10)
(on b11 b7)
(on b12 b3)
(clear b1)
(clear b8)
(clear b9)
(clear b10)
)
(:goal
(and
(on b1 b11)
(on b2 b5)
(on b3 b2)
(on b4 b10)
(on b6 b3)
(on b7 b9)
(on b10 b1)
(on b11 b7))
)
)


