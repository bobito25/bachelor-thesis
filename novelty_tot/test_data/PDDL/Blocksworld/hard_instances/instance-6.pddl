

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a e)
(ontable b)
(ontable c)
(ontable d)
(ontable e)
(on f d)
(clear a)
(clear b)
(clear c)
(clear f)
)
(:goal
(and
(on a f)
(on c a)
(on e d)
(on f b))
)
)


