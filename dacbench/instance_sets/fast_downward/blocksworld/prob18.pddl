

(define (problem BW-rand-11)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 )
(:init
(arm-empty)
(on-table b1)
(on b2 b9)
(on b3 b8)
(on b4 b10)
(on b5 b2)
(on b6 b1)
(on-table b7)
(on b8 b11)
(on b9 b4)
(on b10 b6)
(on b11 b5)
(clear b3)
(clear b7)
)
(:goal
(and
(on b1 b4)
(on b2 b8)
(on b5 b3)
(on b6 b1)
(on b8 b10)
(on b9 b5)
(on b10 b6)
(on b11 b2))
)
)


