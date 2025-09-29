

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a b)
(ontable b)
(ontable c)
(ontable d)
(ontable e)
(on f c)
(clear a)
(clear d)
(clear e)
(clear f)
)
(:goal
(and
(on b f)
(on c d)
(on d b)
(on e c))
)
)


