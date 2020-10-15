

(define (problem BW-rand-10)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 )
(:init
(arm-empty)
(on-table b1)
(on b2 b9)
(on-table b3)
(on b4 b6)
(on-table b5)
(on b6 b10)
(on b7 b4)
(on-table b8)
(on-table b9)
(on-table b10)
(clear b1)
(clear b2)
(clear b3)
(clear b5)
(clear b7)
(clear b8)
)
(:goal
(and
(on b1 b9)
(on b3 b1)
(on b4 b6)
(on b5 b7)
(on b6 b3)
(on b8 b5)
(on b10 b2))
)
)


