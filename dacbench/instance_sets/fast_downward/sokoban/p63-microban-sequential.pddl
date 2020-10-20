;; #####   ####
;; #   ##### .#
;; #       $  ########
;; ###  #### .$    @ #
;;   #  #  #  ####   #
;;   ####  ####  #####

(define (problem p63-microban-sequential)
  (:domain sokoban-sequential)
  (:objects
    dir-down - direction
    dir-left - direction
    dir-right - direction
    dir-up - direction
    player-01 - player
    pos-01-01 - location
    pos-01-02 - location
    pos-01-03 - location
    pos-01-04 - location
    pos-01-05 - location
    pos-01-06 - location
    pos-02-01 - location
    pos-02-02 - location
    pos-02-03 - location
    pos-02-04 - location
    pos-02-05 - location
    pos-02-06 - location
    pos-03-01 - location
    pos-03-02 - location
    pos-03-03 - location
    pos-03-04 - location
    pos-03-05 - location
    pos-03-06 - location
    pos-04-01 - location
    pos-04-02 - location
    pos-04-03 - location
    pos-04-04 - location
    pos-04-05 - location
    pos-04-06 - location
    pos-05-01 - location
    pos-05-02 - location
    pos-05-03 - location
    pos-05-04 - location
    pos-05-05 - location
    pos-05-06 - location
    pos-06-01 - location
    pos-06-02 - location
    pos-06-03 - location
    pos-06-04 - location
    pos-06-05 - location
    pos-06-06 - location
    pos-07-01 - location
    pos-07-02 - location
    pos-07-03 - location
    pos-07-04 - location
    pos-07-05 - location
    pos-07-06 - location
    pos-08-01 - location
    pos-08-02 - location
    pos-08-03 - location
    pos-08-04 - location
    pos-08-05 - location
    pos-08-06 - location
    pos-09-01 - location
    pos-09-02 - location
    pos-09-03 - location
    pos-09-04 - location
    pos-09-05 - location
    pos-09-06 - location
    pos-10-01 - location
    pos-10-02 - location
    pos-10-03 - location
    pos-10-04 - location
    pos-10-05 - location
    pos-10-06 - location
    pos-11-01 - location
    pos-11-02 - location
    pos-11-03 - location
    pos-11-04 - location
    pos-11-05 - location
    pos-11-06 - location
    pos-12-01 - location
    pos-12-02 - location
    pos-12-03 - location
    pos-12-04 - location
    pos-12-05 - location
    pos-12-06 - location
    pos-13-01 - location
    pos-13-02 - location
    pos-13-03 - location
    pos-13-04 - location
    pos-13-05 - location
    pos-13-06 - location
    pos-14-01 - location
    pos-14-02 - location
    pos-14-03 - location
    pos-14-04 - location
    pos-14-05 - location
    pos-14-06 - location
    pos-15-01 - location
    pos-15-02 - location
    pos-15-03 - location
    pos-15-04 - location
    pos-15-05 - location
    pos-15-06 - location
    pos-16-01 - location
    pos-16-02 - location
    pos-16-03 - location
    pos-16-04 - location
    pos-16-05 - location
    pos-16-06 - location
    pos-17-01 - location
    pos-17-02 - location
    pos-17-03 - location
    pos-17-04 - location
    pos-17-05 - location
    pos-17-06 - location
    pos-18-01 - location
    pos-18-02 - location
    pos-18-03 - location
    pos-18-04 - location
    pos-18-05 - location
    pos-18-06 - location
    pos-19-01 - location
    pos-19-02 - location
    pos-19-03 - location
    pos-19-04 - location
    pos-19-05 - location
    pos-19-06 - location
    stone-01 - stone
    stone-02 - stone
  )
  (:init
    (IS-GOAL pos-11-02)
    (IS-GOAL pos-11-04)
    (IS-NONGOAL pos-01-01)
    (IS-NONGOAL pos-01-02)
    (IS-NONGOAL pos-01-03)
    (IS-NONGOAL pos-01-04)
    (IS-NONGOAL pos-01-05)
    (IS-NONGOAL pos-01-06)
    (IS-NONGOAL pos-02-01)
    (IS-NONGOAL pos-02-02)
    (IS-NONGOAL pos-02-03)
    (IS-NONGOAL pos-02-04)
    (IS-NONGOAL pos-02-05)
    (IS-NONGOAL pos-02-06)
    (IS-NONGOAL pos-03-01)
    (IS-NONGOAL pos-03-02)
    (IS-NONGOAL pos-03-03)
    (IS-NONGOAL pos-03-04)
    (IS-NONGOAL pos-03-05)
    (IS-NONGOAL pos-03-06)
    (IS-NONGOAL pos-04-01)
    (IS-NONGOAL pos-04-02)
    (IS-NONGOAL pos-04-03)
    (IS-NONGOAL pos-04-04)
    (IS-NONGOAL pos-04-05)
    (IS-NONGOAL pos-04-06)
    (IS-NONGOAL pos-05-01)
    (IS-NONGOAL pos-05-02)
    (IS-NONGOAL pos-05-03)
    (IS-NONGOAL pos-05-04)
    (IS-NONGOAL pos-05-05)
    (IS-NONGOAL pos-05-06)
    (IS-NONGOAL pos-06-01)
    (IS-NONGOAL pos-06-02)
    (IS-NONGOAL pos-06-03)
    (IS-NONGOAL pos-06-04)
    (IS-NONGOAL pos-06-05)
    (IS-NONGOAL pos-06-06)
    (IS-NONGOAL pos-07-01)
    (IS-NONGOAL pos-07-02)
    (IS-NONGOAL pos-07-03)
    (IS-NONGOAL pos-07-04)
    (IS-NONGOAL pos-07-05)
    (IS-NONGOAL pos-07-06)
    (IS-NONGOAL pos-08-01)
    (IS-NONGOAL pos-08-02)
    (IS-NONGOAL pos-08-03)
    (IS-NONGOAL pos-08-04)
    (IS-NONGOAL pos-08-05)
    (IS-NONGOAL pos-08-06)
    (IS-NONGOAL pos-09-01)
    (IS-NONGOAL pos-09-02)
    (IS-NONGOAL pos-09-03)
    (IS-NONGOAL pos-09-04)
    (IS-NONGOAL pos-09-05)
    (IS-NONGOAL pos-09-06)
    (IS-NONGOAL pos-10-01)
    (IS-NONGOAL pos-10-02)
    (IS-NONGOAL pos-10-03)
    (IS-NONGOAL pos-10-04)
    (IS-NONGOAL pos-10-05)
    (IS-NONGOAL pos-10-06)
    (IS-NONGOAL pos-11-01)
    (IS-NONGOAL pos-11-03)
    (IS-NONGOAL pos-11-05)
    (IS-NONGOAL pos-11-06)
    (IS-NONGOAL pos-12-01)
    (IS-NONGOAL pos-12-02)
    (IS-NONGOAL pos-12-03)
    (IS-NONGOAL pos-12-04)
    (IS-NONGOAL pos-12-05)
    (IS-NONGOAL pos-12-06)
    (IS-NONGOAL pos-13-01)
    (IS-NONGOAL pos-13-02)
    (IS-NONGOAL pos-13-03)
    (IS-NONGOAL pos-13-04)
    (IS-NONGOAL pos-13-05)
    (IS-NONGOAL pos-13-06)
    (IS-NONGOAL pos-14-01)
    (IS-NONGOAL pos-14-02)
    (IS-NONGOAL pos-14-03)
    (IS-NONGOAL pos-14-04)
    (IS-NONGOAL pos-14-05)
    (IS-NONGOAL pos-14-06)
    (IS-NONGOAL pos-15-01)
    (IS-NONGOAL pos-15-02)
    (IS-NONGOAL pos-15-03)
    (IS-NONGOAL pos-15-04)
    (IS-NONGOAL pos-15-05)
    (IS-NONGOAL pos-15-06)
    (IS-NONGOAL pos-16-01)
    (IS-NONGOAL pos-16-02)
    (IS-NONGOAL pos-16-03)
    (IS-NONGOAL pos-16-04)
    (IS-NONGOAL pos-16-05)
    (IS-NONGOAL pos-16-06)
    (IS-NONGOAL pos-17-01)
    (IS-NONGOAL pos-17-02)
    (IS-NONGOAL pos-17-03)
    (IS-NONGOAL pos-17-04)
    (IS-NONGOAL pos-17-05)
    (IS-NONGOAL pos-17-06)
    (IS-NONGOAL pos-18-01)
    (IS-NONGOAL pos-18-02)
    (IS-NONGOAL pos-18-03)
    (IS-NONGOAL pos-18-04)
    (IS-NONGOAL pos-18-05)
    (IS-NONGOAL pos-18-06)
    (IS-NONGOAL pos-19-01)
    (IS-NONGOAL pos-19-02)
    (IS-NONGOAL pos-19-03)
    (IS-NONGOAL pos-19-04)
    (IS-NONGOAL pos-19-05)
    (IS-NONGOAL pos-19-06)
    (MOVE-DIR pos-01-05 pos-01-06 dir-down)
    (MOVE-DIR pos-01-05 pos-02-05 dir-right)
    (MOVE-DIR pos-01-06 pos-01-05 dir-up)
    (MOVE-DIR pos-01-06 pos-02-06 dir-right)
    (MOVE-DIR pos-02-02 pos-02-03 dir-down)
    (MOVE-DIR pos-02-02 pos-03-02 dir-right)
    (MOVE-DIR pos-02-03 pos-02-02 dir-up)
    (MOVE-DIR pos-02-03 pos-03-03 dir-right)
    (MOVE-DIR pos-02-05 pos-01-05 dir-left)
    (MOVE-DIR pos-02-05 pos-02-06 dir-down)
    (MOVE-DIR pos-02-06 pos-01-06 dir-left)
    (MOVE-DIR pos-02-06 pos-02-05 dir-up)
    (MOVE-DIR pos-03-02 pos-02-02 dir-left)
    (MOVE-DIR pos-03-02 pos-03-03 dir-down)
    (MOVE-DIR pos-03-02 pos-04-02 dir-right)
    (MOVE-DIR pos-03-03 pos-02-03 dir-left)
    (MOVE-DIR pos-03-03 pos-03-02 dir-up)
    (MOVE-DIR pos-03-03 pos-04-03 dir-right)
    (MOVE-DIR pos-04-02 pos-03-02 dir-left)
    (MOVE-DIR pos-04-02 pos-04-03 dir-down)
    (MOVE-DIR pos-04-03 pos-03-03 dir-left)
    (MOVE-DIR pos-04-03 pos-04-02 dir-up)
    (MOVE-DIR pos-04-03 pos-04-04 dir-down)
    (MOVE-DIR pos-04-03 pos-05-03 dir-right)
    (MOVE-DIR pos-04-04 pos-04-03 dir-up)
    (MOVE-DIR pos-04-04 pos-04-05 dir-down)
    (MOVE-DIR pos-04-04 pos-05-04 dir-right)
    (MOVE-DIR pos-04-05 pos-04-04 dir-up)
    (MOVE-DIR pos-04-05 pos-05-05 dir-right)
    (MOVE-DIR pos-05-03 pos-04-03 dir-left)
    (MOVE-DIR pos-05-03 pos-05-04 dir-down)
    (MOVE-DIR pos-05-03 pos-06-03 dir-right)
    (MOVE-DIR pos-05-04 pos-04-04 dir-left)
    (MOVE-DIR pos-05-04 pos-05-03 dir-up)
    (MOVE-DIR pos-05-04 pos-05-05 dir-down)
    (MOVE-DIR pos-05-05 pos-04-05 dir-left)
    (MOVE-DIR pos-05-05 pos-05-04 dir-up)
    (MOVE-DIR pos-06-01 pos-07-01 dir-right)
    (MOVE-DIR pos-06-03 pos-05-03 dir-left)
    (MOVE-DIR pos-06-03 pos-07-03 dir-right)
    (MOVE-DIR pos-07-01 pos-06-01 dir-left)
    (MOVE-DIR pos-07-01 pos-08-01 dir-right)
    (MOVE-DIR pos-07-03 pos-06-03 dir-left)
    (MOVE-DIR pos-07-03 pos-08-03 dir-right)
    (MOVE-DIR pos-07-05 pos-07-06 dir-down)
    (MOVE-DIR pos-07-05 pos-08-05 dir-right)
    (MOVE-DIR pos-07-06 pos-07-05 dir-up)
    (MOVE-DIR pos-07-06 pos-08-06 dir-right)
    (MOVE-DIR pos-08-01 pos-07-01 dir-left)
    (MOVE-DIR pos-08-03 pos-07-03 dir-left)
    (MOVE-DIR pos-08-03 pos-09-03 dir-right)
    (MOVE-DIR pos-08-05 pos-07-05 dir-left)
    (MOVE-DIR pos-08-05 pos-08-06 dir-down)
    (MOVE-DIR pos-08-06 pos-07-06 dir-left)
    (MOVE-DIR pos-08-06 pos-08-05 dir-up)
    (MOVE-DIR pos-09-03 pos-08-03 dir-left)
    (MOVE-DIR pos-09-03 pos-10-03 dir-right)
    (MOVE-DIR pos-10-02 pos-10-03 dir-down)
    (MOVE-DIR pos-10-02 pos-11-02 dir-right)
    (MOVE-DIR pos-10-03 pos-09-03 dir-left)
    (MOVE-DIR pos-10-03 pos-10-02 dir-up)
    (MOVE-DIR pos-10-03 pos-10-04 dir-down)
    (MOVE-DIR pos-10-03 pos-11-03 dir-right)
    (MOVE-DIR pos-10-04 pos-10-03 dir-up)
    (MOVE-DIR pos-10-04 pos-10-05 dir-down)
    (MOVE-DIR pos-10-04 pos-11-04 dir-right)
    (MOVE-DIR pos-10-05 pos-10-04 dir-up)
    (MOVE-DIR pos-10-05 pos-11-05 dir-right)
    (MOVE-DIR pos-11-02 pos-10-02 dir-left)
    (MOVE-DIR pos-11-02 pos-11-03 dir-down)
    (MOVE-DIR pos-11-03 pos-10-03 dir-left)
    (MOVE-DIR pos-11-03 pos-11-02 dir-up)
    (MOVE-DIR pos-11-03 pos-11-04 dir-down)
    (MOVE-DIR pos-11-04 pos-10-04 dir-left)
    (MOVE-DIR pos-11-04 pos-11-03 dir-up)
    (MOVE-DIR pos-11-04 pos-11-05 dir-down)
    (MOVE-DIR pos-11-04 pos-12-04 dir-right)
    (MOVE-DIR pos-11-05 pos-10-05 dir-left)
    (MOVE-DIR pos-11-05 pos-11-04 dir-up)
    (MOVE-DIR pos-12-04 pos-11-04 dir-left)
    (MOVE-DIR pos-12-04 pos-13-04 dir-right)
    (MOVE-DIR pos-13-01 pos-13-02 dir-down)
    (MOVE-DIR pos-13-01 pos-14-01 dir-right)
    (MOVE-DIR pos-13-02 pos-13-01 dir-up)
    (MOVE-DIR pos-13-02 pos-14-02 dir-right)
    (MOVE-DIR pos-13-04 pos-12-04 dir-left)
    (MOVE-DIR pos-13-04 pos-14-04 dir-right)
    (MOVE-DIR pos-13-06 pos-14-06 dir-right)
    (MOVE-DIR pos-14-01 pos-13-01 dir-left)
    (MOVE-DIR pos-14-01 pos-14-02 dir-down)
    (MOVE-DIR pos-14-01 pos-15-01 dir-right)
    (MOVE-DIR pos-14-02 pos-13-02 dir-left)
    (MOVE-DIR pos-14-02 pos-14-01 dir-up)
    (MOVE-DIR pos-14-02 pos-15-02 dir-right)
    (MOVE-DIR pos-14-04 pos-13-04 dir-left)
    (MOVE-DIR pos-14-04 pos-15-04 dir-right)
    (MOVE-DIR pos-14-06 pos-13-06 dir-left)
    (MOVE-DIR pos-15-01 pos-14-01 dir-left)
    (MOVE-DIR pos-15-01 pos-15-02 dir-down)
    (MOVE-DIR pos-15-01 pos-16-01 dir-right)
    (MOVE-DIR pos-15-02 pos-14-02 dir-left)
    (MOVE-DIR pos-15-02 pos-15-01 dir-up)
    (MOVE-DIR pos-15-02 pos-16-02 dir-right)
    (MOVE-DIR pos-15-04 pos-14-04 dir-left)
    (MOVE-DIR pos-15-04 pos-16-04 dir-right)
    (MOVE-DIR pos-16-01 pos-15-01 dir-left)
    (MOVE-DIR pos-16-01 pos-16-02 dir-down)
    (MOVE-DIR pos-16-01 pos-17-01 dir-right)
    (MOVE-DIR pos-16-02 pos-15-02 dir-left)
    (MOVE-DIR pos-16-02 pos-16-01 dir-up)
    (MOVE-DIR pos-16-02 pos-17-02 dir-right)
    (MOVE-DIR pos-16-04 pos-15-04 dir-left)
    (MOVE-DIR pos-16-04 pos-16-05 dir-down)
    (MOVE-DIR pos-16-04 pos-17-04 dir-right)
    (MOVE-DIR pos-16-05 pos-16-04 dir-up)
    (MOVE-DIR pos-16-05 pos-17-05 dir-right)
    (MOVE-DIR pos-17-01 pos-16-01 dir-left)
    (MOVE-DIR pos-17-01 pos-17-02 dir-down)
    (MOVE-DIR pos-17-01 pos-18-01 dir-right)
    (MOVE-DIR pos-17-02 pos-16-02 dir-left)
    (MOVE-DIR pos-17-02 pos-17-01 dir-up)
    (MOVE-DIR pos-17-02 pos-18-02 dir-right)
    (MOVE-DIR pos-17-04 pos-16-04 dir-left)
    (MOVE-DIR pos-17-04 pos-17-05 dir-down)
    (MOVE-DIR pos-17-04 pos-18-04 dir-right)
    (MOVE-DIR pos-17-05 pos-16-05 dir-left)
    (MOVE-DIR pos-17-05 pos-17-04 dir-up)
    (MOVE-DIR pos-17-05 pos-18-05 dir-right)
    (MOVE-DIR pos-18-01 pos-17-01 dir-left)
    (MOVE-DIR pos-18-01 pos-18-02 dir-down)
    (MOVE-DIR pos-18-01 pos-19-01 dir-right)
    (MOVE-DIR pos-18-02 pos-17-02 dir-left)
    (MOVE-DIR pos-18-02 pos-18-01 dir-up)
    (MOVE-DIR pos-18-02 pos-19-02 dir-right)
    (MOVE-DIR pos-18-04 pos-17-04 dir-left)
    (MOVE-DIR pos-18-04 pos-18-05 dir-down)
    (MOVE-DIR pos-18-05 pos-17-05 dir-left)
    (MOVE-DIR pos-18-05 pos-18-04 dir-up)
    (MOVE-DIR pos-19-01 pos-18-01 dir-left)
    (MOVE-DIR pos-19-01 pos-19-02 dir-down)
    (MOVE-DIR pos-19-02 pos-18-02 dir-left)
    (MOVE-DIR pos-19-02 pos-19-01 dir-up)
    (at player-01 pos-17-04)
    (at stone-01 pos-09-03)
    (at stone-02 pos-12-04)
    (clear pos-01-05)
    (clear pos-01-06)
    (clear pos-02-02)
    (clear pos-02-03)
    (clear pos-02-05)
    (clear pos-02-06)
    (clear pos-03-02)
    (clear pos-03-03)
    (clear pos-04-02)
    (clear pos-04-03)
    (clear pos-04-04)
    (clear pos-04-05)
    (clear pos-05-03)
    (clear pos-05-04)
    (clear pos-05-05)
    (clear pos-06-01)
    (clear pos-06-03)
    (clear pos-07-01)
    (clear pos-07-03)
    (clear pos-07-05)
    (clear pos-07-06)
    (clear pos-08-01)
    (clear pos-08-03)
    (clear pos-08-05)
    (clear pos-08-06)
    (clear pos-10-02)
    (clear pos-10-03)
    (clear pos-10-04)
    (clear pos-10-05)
    (clear pos-11-02)
    (clear pos-11-03)
    (clear pos-11-04)
    (clear pos-11-05)
    (clear pos-13-01)
    (clear pos-13-02)
    (clear pos-13-04)
    (clear pos-13-06)
    (clear pos-14-01)
    (clear pos-14-02)
    (clear pos-14-04)
    (clear pos-14-06)
    (clear pos-15-01)
    (clear pos-15-02)
    (clear pos-15-04)
    (clear pos-16-01)
    (clear pos-16-02)
    (clear pos-16-04)
    (clear pos-16-05)
    (clear pos-17-01)
    (clear pos-17-02)
    (clear pos-17-05)
    (clear pos-18-01)
    (clear pos-18-02)
    (clear pos-18-04)
    (clear pos-18-05)
    (clear pos-19-01)
    (clear pos-19-02)
    (= (total-cost) 0)
  )
  (:goal (and
    (at-goal stone-01)
    (at-goal stone-02)
  ))
  (:metric minimize (total-cost))
)
