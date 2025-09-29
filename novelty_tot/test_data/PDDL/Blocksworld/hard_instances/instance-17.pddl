

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a f)
(on b d)
(ontable c)
(ontable d)
(ontable e)
(ontable f)
(clear a)
(clear b)
(clear c)
(clear e)
)
(:goal
(and
(on a c)
(on b e)
(on c f)
(on d b)
(on e a))
)
)


