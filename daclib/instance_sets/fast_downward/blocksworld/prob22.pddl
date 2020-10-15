

(define (problem BW-rand-12)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 )
(:init
(arm-empty)
(on b1 b9)
(on b2 b3)
(on b3 b10)
(on b4 b2)
(on-table b5)
(on-table b6)
(on-table b7)
(on b8 b6)
(on b9 b7)
(on b10 b8)
(on b11 b5)
(on b12 b1)
(clear b4)
(clear b11)
(clear b12)
)
(:goal
(and
(on b2 b1)
(on b3 b4)
(on b4 b2)
(on b5 b7)
(on b6 b10)
(on b8 b9)
(on b9 b6)
(on b10 b11)
(on b11 b5))
)
)


