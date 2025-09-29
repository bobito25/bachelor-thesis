

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a d)
(on b c)
(on c f)
(ontable d)
(on e b)
(ontable f)
(clear a)
(clear e)
)
(:goal
(and
(on a d)
(on b f)
(on e c)
(on f a))
)
)


