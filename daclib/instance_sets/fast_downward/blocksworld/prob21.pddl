

(define (problem BW-rand-12)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 )
(:init
(arm-empty)
(on-table b1)
(on b2 b3)
(on b3 b4)
(on-table b4)
(on-table b5)
(on b6 b2)
(on b7 b6)
(on b8 b10)
(on b9 b5)
(on b10 b12)
(on b11 b9)
(on b12 b7)
(clear b1)
(clear b8)
(clear b11)
)
(:goal
(and
(on b2 b11)
(on b3 b1)
(on b4 b6)
(on b5 b10)
(on b6 b5)
(on b8 b9)
(on b9 b4)
(on b10 b3)
(on b11 b12)
(on b12 b8))
)
)


