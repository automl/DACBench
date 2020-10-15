

(define (problem BW-rand-10)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 )
(:init
(arm-empty)
(on b1 b4)
(on-table b2)
(on-table b3)
(on b4 b2)
(on b5 b10)
(on-table b6)
(on b7 b5)
(on-table b8)
(on b9 b7)
(on b10 b6)
(clear b1)
(clear b3)
(clear b8)
(clear b9)
)
(:goal
(and
(on b1 b7)
(on b3 b10)
(on b5 b2)
(on b6 b9)
(on b7 b5)
(on b9 b4)
(on b10 b1))
)
)


