

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a b)
(on b d)
(ontable c)
(ontable d)
(on e a)
(on f c)
(clear e)
(clear f)
)
(:goal
(and
(on c d)
(on d b)
(on e c)
(on f e))
)
)


