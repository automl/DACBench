

(define (problem BW-rand-11)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 )
(:init
(arm-empty)
(on b1 b11)
(on b2 b10)
(on b3 b1)
(on b4 b3)
(on-table b5)
(on b6 b5)
(on-table b7)
(on-table b8)
(on b9 b2)
(on-table b10)
(on b11 b9)
(clear b4)
(clear b6)
(clear b7)
(clear b8)
)
(:goal
(and
(on b1 b7)
(on b2 b1)
(on b3 b4)
(on b4 b8)
(on b5 b2)
(on b8 b11)
(on b9 b6)
(on b10 b9)
(on b11 b5))
)
)


