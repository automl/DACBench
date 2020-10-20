;; #####
;; #@$.#
;; #####

(define (problem p44-microban-sequential)
  (:domain sokoban-sequential)
  (:objects
    dir-down - direction
    dir-left - direction
    dir-right - direction
    dir-up - direction
    player-01 - player
    pos-1-1 - location
    pos-1-2 - location
    pos-1-3 - location
    pos-2-1 - location
    pos-2-2 - location
    pos-2-3 - location
    pos-3-1 - location
    pos-3-2 - location
    pos-3-3 - location
    pos-4-1 - location
    pos-4-2 - location
    pos-4-3 - location
    pos-5-1 - location
    pos-5-2 - location
    pos-5-3 - location
    stone-01 - stone
  )
  (:init
    (IS-GOAL pos-4-2)
    (IS-NONGOAL pos-1-1)
    (IS-NONGOAL pos-1-2)
    (IS-NONGOAL pos-1-3)
    (IS-NONGOAL pos-2-1)
    (IS-NONGOAL pos-2-2)
    (IS-NONGOAL pos-2-3)
    (IS-NONGOAL pos-3-1)
    (IS-NONGOAL pos-3-2)
    (IS-NONGOAL pos-3-3)
    (IS-NONGOAL pos-4-1)
    (IS-NONGOAL pos-4-3)
    (IS-NONGOAL pos-5-1)
    (IS-NONGOAL pos-5-2)
    (IS-NONGOAL pos-5-3)
    (MOVE-DIR pos-2-2 pos-3-2 dir-right)
    (MOVE-DIR pos-3-2 pos-2-2 dir-left)
    (MOVE-DIR pos-3-2 pos-4-2 dir-right)
    (MOVE-DIR pos-4-2 pos-3-2 dir-left)
    (at player-01 pos-2-2)
    (at stone-01 pos-3-2)
    (clear pos-4-2)
    (= (total-cost) 0)
  )
  (:goal (and
    (at-goal stone-01)
  ))
  (:metric minimize (total-cost))
)
