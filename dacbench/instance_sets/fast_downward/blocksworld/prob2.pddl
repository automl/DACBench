

(define (problem BW-rand-10)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 )
(:init
(arm-empty)
(on-table b1)
(on b2 b7)
(on-table b3)
(on b4 b3)
(on b5 b10)
(on b6 b9)
(on b7 b6)
(on b8 b2)
(on b9 b1)
(on b10 b8)
(clear b4)
(clear b5)
)
(:goal
(and
(on b4 b7)
(on b5 b8)
(on b8 b1)
(on b9 b4)
(on b10 b2))
)
)


